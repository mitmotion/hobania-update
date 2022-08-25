use arbalest::sync::{Strong, Frail};
use core::{
    fmt,
    future::Future,
    marker::Unsize,
    ops::{CoerceUnsized, DispatchFromDyn},
    pin::Pin,
    sync::{atomic::{AtomicBool, Ordering}, Exclusive},
    task,
};
use executors::{
    crossbeam_workstealing_pool::ThreadPool as ThreadPool_,
    parker::{LargeThreadData, StaticParker},
    Executor,
};
pub use executors::parker::large;
use hashbrown::{hash_map::Entry, HashMap};
use pin_project_lite::pin_project;
// use rayon::ThreadPool;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::{error, warn/* , Instrument */};

pub type ThreadPool = ThreadPool_<StaticParker<LargeThreadData>>;

/// Provides a Wrapper around rayon threadpool to execute slow-jobs.
/// slow means, the job doesn't need to not complete within the same tick.
/// DO NOT USE I/O blocking jobs, but only CPU heavy jobs.
/// Jobs run here, will reduce the ammount of threads rayon can use during the
/// main tick.
///
/// ## Configuration
/// This Pool allows you to configure certain names of jobs and assign them a
/// maximum number of threads # Example
/// Your system has 16 cores, you assign 12 cores for slow-jobs.
/// Then you can configure all jobs with the name `CHUNK_GENERATOR` to spawn on
/// max 50% (6 = cores)
///
/// ## Spawn Order
/// - At least 1 job of a configuration is allowed to run if global limit isn't
///   hit.
/// - remaining capacities are spread in relation to their limit. e.g. a
///   configuration with double the limit will be sheduled to spawn double the
///   tasks, starting by a round robin.
///
/// ## States
/// - queued
/// - spawned
/// - started
/// - finished
/// ```
/// # use veloren_common::slowjob::SlowJobPool;
/// # use std::sync::Arc;
///
/// let threadpool = rayon::ThreadPoolBuilder::new()
///     .num_threads(16)
///     .build()
///     .unwrap();
/// let pool = SlowJobPool::new(3, 10, Arc::new(threadpool));
/// pool.configure("CHUNK_GENERATOR", |n| n / 2);
/// pool.spawn("CHUNK_GENERATOR", move || println!("this is a job"));
/// ```
// #[derive(Clone)]
pub struct SlowJobPool {
    internal: Arc<Mutex<InternalSlowJobPool>>,
    threadpool: ThreadPool,
}

impl Drop for SlowJobPool {
    fn drop(&mut self) {
        self.threadpool.shutdown_borrowed();
    }
}

type Name = /*String*/&'static str;

#[derive(Debug)]
pub struct SlowJob {
    task: Pin<Frail<Queue>>,
}

// impl<T: ?Sized + Unsize<U> + CoerceUnsized<U>, U: ?Sized> CoerceUnsized<Task<U>> for Task<T> {}

struct InternalSlowJobPool {
    cur_slot: usize,
    queue: HashMap<Name, VecDeque<Pin<Strong<Queue>>>>,
    configs: HashMap<Name, Config>,
    last_spawned_configs: Vec<Name>,
    global_spawned_and_running: u64,
    global_limit: u64,
    jobs_metrics_cnt: usize,
    jobs_metrics: HashMap<Name, Vec<JobMetrics>>,
}

#[derive(Debug)]
struct Config {
    local_limit: u64,
    local_spawned_and_running: u64,
}

pin_project! {
    struct Task<F: ?Sized> {
        queue_created: Instant,
        // Has this task been canceled?
        is_canceled: AtomicBool,
        #[pin]
        // The actual task future.
        task: Exclusive<F>,
    }
}

impl<F: Future + ?Sized> Future for Task<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Self::Output> {
        self.project().task.poll(cx)
    }
}

/// NOTE: Should be FnOnce, but can't because there's no easy way to run an FnOnce function on an
/// Arc even if [try_unwrap] would work.  We could write non-safe code to do this, but it probably
/// isn't worth it.
type Queue = Task<dyn /*FnMut()*/Future<Output=()> + Send + 'static>;

pub struct JobMetrics {
    pub queue_created: Instant,
    pub execution_start: Instant,
    pub execution_end: Instant,
}

impl<F> Task<F> {
    fn new(f: F) -> Task<impl /*FnMut()*/Future<Output=F::Output> + Send + 'static>
        where F: /*FnOnce()*/Future + Send + 'static
    {
        let queue_created = Instant::now();
        /* let mut f = Some(f); */
        Task {
            queue_created,
            is_canceled: AtomicBool::new(false),
            task: Exclusive::new(/* move || {
                // Working around not being able to call FnOnce in an Arc.
                if let Some(f) = f.take() {
                    f();
                }
            }*/f),
        }
    }
}

impl InternalSlowJobPool {
    pub fn new(
        global_limit: u64,
        jobs_metrics_cnt: usize,
    ) -> Self {
        Self {
            queue: HashMap::new(),
            configs: HashMap::new(),
            cur_slot: 0,
            last_spawned_configs: Vec::new(),
            global_spawned_and_running: 0,
            global_limit: global_limit.max(1),
            jobs_metrics_cnt,
            jobs_metrics: HashMap::new(),
        }
    }

    /// returns order of configuration which are queued next
    fn calc_queued_order(
        &self,
        mut queued: HashMap<&Name, u64>,
        mut limit: usize,
    ) -> Vec<Name> {
        let mut roundrobin = self.last_spawned_configs.clone();
        let mut result = vec![];
        /* let spawned = self
            .configs
            .iter()
            .map(|(n, c)| (n, c.local_spawned_and_running))
            .collect::<HashMap<_, u64>>(); */
        let spawned = &self.configs;
        let mut queried_capped = self
            .configs
            .iter()
            .map(|(n, c)| {
                (
                    n,
                    queued
                        .get(&n)
                        .cloned()
                        .unwrap_or(0)
                        .min(c.local_limit - c.local_spawned_and_running),
                )
            })
            .collect::<HashMap<_, _>>();
        // grab all configs that are queued and not running. in roundrobin order
        for n in roundrobin.clone().into_iter() {
            if let Some(c) = queued.get_mut(&n) {
                if *c > 0 && spawned.get(&n).map(|c| c.local_spawned_and_running).unwrap_or(0) == 0 {
                    result.push(n.clone());
                    *c -= 1;
                    limit -= 1;
                    queried_capped.get_mut(&n).map(|v| *v -= 1);
                    roundrobin
                        .iter()
                        .position(|e| e == &n)
                        .map(|i| roundrobin.remove(i));
                    roundrobin.push(n);
                    if limit == 0 {
                        return result;
                    }
                }
            }
        }
        //schedule rest based on their possible limites, don't use round robin here
        let total_limit = queried_capped.values().sum::<u64>() as f32;
        if total_limit < f32::EPSILON {
            return result;
        }
        let mut spawn_rates = queried_capped
            .iter()
            .map(|(&n, l)| (n, ((*l as f32 * limit as f32) / total_limit).min(*l as f32)))
            .collect::<Vec<_>>();
        while limit > 0 {
            spawn_rates.sort_by(|(_, a), (_, b)| {
                if b < a {
                    core::cmp::Ordering::Less
                } else if (b - a).abs() < f32::EPSILON {
                    core::cmp::Ordering::Equal
                } else {
                    core::cmp::Ordering::Greater
                }
            });
            match spawn_rates.first_mut() {
                Some((n, r)) => {
                    if *r > f32::EPSILON {
                        result.push(n.to_owned());
                        limit -= 1;
                        *r -= 1.0;
                    } else {
                        break;
                    }
                },
                None => break,
            }
        }
        result
    }

    fn can_spawn(&self, name: &Name) -> bool {
        let queued = self
            .queue
            .iter()
            .map(|(n, m)| (n, m.len() as u64))
            .collect::<HashMap<_, u64>>();
        let mut to_be_queued = queued.clone();
        let name = name.to_owned();
        *to_be_queued.entry(&name).or_default() += 1;
        let limit = (self.global_limit - self.global_spawned_and_running) as usize;
        // calculate to_be_queued first
        let to_be_queued_order = self.calc_queued_order(to_be_queued, limit);
        let queued_order = self.calc_queued_order(queued, limit);
        // if its queued one time more then its okay to spawn
        let to_be_queued_cnt = to_be_queued_order
            .into_iter()
            .filter(|n| n == &name)
            .count();
        let queued_cnt = queued_order.into_iter().filter(|n| n == &name).count();
        to_be_queued_cnt > queued_cnt
    }

    fn spawn<F>(&mut self, slowjob: &SlowJobPool, push_back: bool, name: &Name, f: F) -> SlowJob
    where
        F: /*FnOnce()*/Future<Output=()> + Send + 'static,
    {
        // let f = f.instrument(tracing::info_span!("{}", name));
        let queue: Pin<Strong<Queue>> = Strong::pin(Task::new(f));
        let mut deque = self.queue
            .entry(name.to_owned())
            .or_default();
        let job = SlowJob {
            task: Strong::pin_downgrade(&queue)
        };
        if push_back {
            deque.push_back(queue);
        } else {
            deque.push_front(queue);
        }
        debug_assert!(
            self.configs.contains_key(name),
            "Can't spawn unconfigured task!"
        );
        //spawn already queued
        self.spawn_queued(slowjob);
        job
    }

    fn finish(&mut self, name: &Name, metrics: JobMetrics) {
        let metric = self.jobs_metrics.entry(name.to_owned()).or_default();

        if metric.len() < self.jobs_metrics_cnt {
            metric.push(metrics);
        }
        self.global_spawned_and_running -= 1;
        if let Some(c) = self.configs.get_mut(name) {
            c.local_spawned_and_running -= 1;
        } else {
            warn!(?name, "sync_maintain on a no longer existing config");
        }
    }

    /// NOTE: This does not spawn the job directly, but it *does* increment cur_slot and the local
    /// and global task counters, so make sure to actually finish the returned jobs if you consume
    /// the iterator, or the position in the queue may be off!
    #[must_use = "Remember to actually use the returned jobs if you consume the iterator."]
    fn next_jobs<'a>(&'a mut self) -> impl Iterator<Item = (Name, Pin<Strong<Queue>>)> + 'a {
        let queued = &mut self.queue;
        let configs = &mut self.configs;
        let global_spawned_and_running = &mut self.global_spawned_and_running;

        let cur_slot = &mut self.cur_slot;
        let num_slots = self.last_spawned_configs.len();
        let jobs_limit = self.global_limit.saturating_sub(*global_spawned_and_running) as usize;

        let queued_order = self.last_spawned_configs.iter().cycle().skip(*cur_slot).take(num_slots);
        queued_order
            // NOTE: num_slots > 0, because queued_order can only yield up to num_slots elements.
            .inspect(move |_| { *cur_slot = (*cur_slot + 1) % num_slots; })
            .filter_map(move |name| {
                let deque = queued.get_mut(name)?;
                let config = configs.get_mut(name)?;
                if /* config.local_spawned_and_running < config.local_limit*/true {
                    let task = deque.pop_front()?;
                    config.local_spawned_and_running += 1;
                    *global_spawned_and_running += 1;
                    Some((name.to_owned(), task))
                } else {
                    None
                }
            })
            .take(jobs_limit)
    }

    /// Spawn tasks in the threadpool, in round-robin order.
    ///
    /// NOTE: Do *not* call this from an existing thread in the threadpool.
    fn spawn_queued(&mut self, slowjob: &SlowJobPool) {
        /* let old_running = self.global_spawned_and_running; */
        while self.next_jobs().map(|task| slowjob.spawn_in_threadpool(task)).count() != 0 {}
        /* let total_spawned = (self.global_spawned_and_running - old_running) as usize;
        self.cur_slot = (initial_slot + total_spawned) % num_slots;
        self.cur_slot %= num_slots; */
        /* let queued = self
            .queue
            .iter_mut();
            /* .iter();
            .map(|(n, m)| (n, m.len() as u64))
            .collect::<HashMap<_, u64>>();
        let limit = self.global_limit as usize;
        let queued_order = self.calc_queued_order(queued, limit); */

        let queued_order = queued;
        for (name, deque) in queued_order/*.into_iter()*/.take(self.global_limit.saturating_sub(self.global_spawned_and_running) as usize) {
            /* match self.queue.get_mut(&name) {
                Some(deque) => */match deque.pop_front() {
                    Some(queue) => {
                        //fire
                        self.global_spawned_and_running += 1;
                        self.configs
                            .get_mut(&queue.name)
                            .expect("cannot fire a unconfigured job")
                            .local_spawned_and_running += 1;
                        self.last_spawned_configs
                            .iter()
                            .position(|e| e == &queue.name)
                            .map(|i| self.last_spawned_configs.remove(i));
                        self.last_spawned_configs.push((&queue.name).to_owned());
                        self.threadpool.spawn(queue.task);
                    },
                    None => /* error!(
                        "internal calculation is wrong, we extected a schedulable job to be \
                         present in the queue"
                    ),*/{}
                }/*,
                None => error!(
                    "internal calculation is wrong, we marked a queue as schedulable which \
                     doesn't exist"
                ),
            } */
        } */

        /* let queued = self
            .queue
            .iter()
            .map(|(n, m)| (n, m.len() as u64))
            .collect::<HashMap<_, u64>>();
        let limit = self.global_limit as usize;
        let queued_order = self.calc_queued_order(queued, limit);
        for name in queued_order.into_iter() {
            match self.queue.get_mut(&name) {
                Some(deque) => match deque.pop_front() {
                    Some(queue) => {
                        //fire
                        self.global_spawned_and_running += 1;
                        self.configs
                            .get_mut(&queue.name)
                            .expect("cannot fire a unconfigured job")
                            .local_spawned_and_running += 1;
                        self.last_spawned_configs
                            .iter()
                            .position(|e| e == &queue.name)
                            .map(|i| self.last_spawned_configs.remove(i));
                        self.last_spawned_configs.push((&queue.name).to_owned());
                        self.threadpool.spawn(queue.task);
                    },
                    None => error!(
                        "internal calculation is wrong, we extected a schedulable job to be \
                         present in the queue"
                    ),
                },
                None => error!(
                    "internal calculation is wrong, we marked a queue as schedulable which \
                     doesn't exist"
                ),
            }
        } */
    }

    pub fn take_metrics(&mut self) -> HashMap<Name, Vec<JobMetrics>> {
        core::mem::take(&mut self.jobs_metrics)
    }
}


impl SlowJob {
    /// TODO: This would be simplified (and perform a bit better) if there existed a structure that
    /// "split" an Arc allocation into two parts, a shared and owned part (with types corresponding
    /// to references to each).  The strong type would not be cloneable and would grant mutable
    /// access to the owned part, and shared access to the shared part; the weak type would be
    /// cloneable, but would only shared access to the shared part, and no access to the owned
    /// part.  This would allow us to share the allocation, without needing to keep track of an
    /// explicit weak pointer count, perform any sort of locking on cancelation, etc.
    /// Unfortunately I cannot find such a type on crates.io, and writing one would be a fairly
    /// involved task, so we defer this for now.
    pub fn cancel(self) -> Result<(), Self> {
        // Correctness of cancellation is a bit subtle, due to wanting to avoid locking the queue
        // more than necessary, iterate over jobs, or introduce a way to access jobs by key.
        //
        // First, we try to upgrade our weak reference to the Arc.  This will fail if the strong
        // reference is currently mutably borrowed, or if the strong side has already been
        // dropped.  Since we never mutably borrow the reference until we're definitely going to
        // run the task, and we only drop the strong side after the task is complete, this is
        // a conservative signal that there's no point in cancelling the task, so this has no
        // false positives.
        let task = Frail::try_pin_upgrade(&self.task).or(Err(self))?;
        // Now that the task is upgraded, any attempt by the strong side to mutably access the
        // task will fail, so it will assume it's been canceled.  This is fine, because we're
        // about to cancel it anyway.
        //
        // Next, we try to signal (monotonically) that the task should be cancelled, by updating
        // the value of canceled atomically to true.  Since this is monotonic, we can use Relaxed
        // here.  It would actually be fine if this signal was lost, since cancellation is always
        // an optimization, but with the current implementation it won't be--the strong side only
        // checks for cancellation after it tries to mutably access the task, which can't happen
        // while the task is "locked" by our weak upgrade, so our write here will always be
        // visible.
        task.is_canceled.store(true, Ordering::Relaxed);
        // Note that we don't bother to check whether the task was already canceled.  Firstly,
        // because we don't care, secondly because even if we did, this function takes ownership of
        // the SlowJob, which contains the only weak reference with the ability to cancel, so no
        // job can be canceled more than once anyway.
        Ok(())
    }
}

impl SlowJobPool {
    pub fn new(global_limit: u64, jobs_metrics_cnt: usize, threadpool: /*Arc<*/ThreadPool/*>*/) -> Self {
        Self {
            internal: Arc::new(Mutex::new(InternalSlowJobPool::new(global_limit, jobs_metrics_cnt))),
            threadpool,
        }
    }

    /// configure a NAME to spawn up to f(n) threads, depending on how many
    /// threads we globally have available
    pub fn configure<F>(&self, name: &Name, f: F)
    where
        F: Fn(u64) -> u64,
    {
        let mut lock = self.internal.lock().expect("lock poisoned while configure");
        let lock = &mut *lock;
        // Make sure not to update already-present config entries, since this can mess up some of
        // the stuff we do that assumes monotonicity.
        if let Entry::Vacant(v) = lock.configs.entry(name.to_owned()) {
            let cnf = Config {
                local_limit: f(lock.global_limit).max(1),
                local_spawned_and_running: 0,
            };
            let cnf = v.insert(cnf);
            // Add cnf into the entry list as many times as its local limit, to ensure that stuff
            // gets run more often if it has more assigned threads.
            lock.last_spawned_configs.resize(lock.last_spawned_configs.len() + /* cnf.local_limit as usize */1, name.to_owned());
        }
    }

    /// Spawn a task in the threadpool.
    ///
    /// This runs the task, and then checks at the end to see if there are any more tasks to run
    /// before returning for good.  In cases with lots of tasks, this may help avoid unnecessary
    /// context switches or extra threads being spawned unintentionally.
    fn spawn_in_threadpool(&self, mut name_task: (Name, Pin<Strong<Queue>>)) {
        let internal = Arc::clone(&self.internal);

        // NOTE: It's important not to use internal until we're in the spawned thread, since the
        // lock is probably currently taken!
        self.threadpool.execute(move || {
            // Repeatedly run until exit; we do things this way to avoid recursion, which might blow
            // our call stack.
            loop {
                let (name, mut task) = name_task;
                let queue_created = task.queue_created;
                // See the [SlowJob::cancel] method for justification for this step's correctness.
                //
                // NOTE: This is not exact because we do it before borrowing the task, but the
                // difference is minor and it makes it easier to assign metrics to canceled tasks
                // (though maybe we don't want to do that?).
                let execution_start = Instant::now();
                if let Some(mut task) = Strong::try_pin_borrow_mut(&mut task)
                    .ok()
                    .filter(|task| !task.is_canceled.load(Ordering::Relaxed)) {
                    // The task was not canceled.
                    //
                    // Run the task in its own scope so perf works correctly.
                    common_base::prof_span_alloc!(_guard, &name);
                    futures::executor::block_on(task.as_mut()/* .instrument({
                        common_base::prof_span!(span, &name);
                        span
                    }) */);
                }
                let execution_end = Instant::now();
                let metrics = JobMetrics {
                    queue_created,
                    execution_start,
                    execution_end,
                };
                // directly maintain the next task afterwards
                let next_task = {
                    // We take the lock in this scope to make sure it's dropped before we
                    // actully launch the next job.
                    let mut lock = internal.lock().expect("slowjob lock poisoned");
                    let lock = &mut *lock;
                    lock.finish(&name, metrics);
                    let mut jobs = lock.next_jobs();
                    jobs.next()
                };
                name_task = if let Some(name_task) = next_task {
                    // We launch the job on the *same* thread, since we're already in the
                    // thread pool.
                    name_task
                } else {
                    // There are no more tasks to run at this time, so we exit the thread in
                    // the threadpool (in theory, it might make sense to yield or spin a few
                    // times or something in case we have more tasks to execute).
                    return;
                };
            }
        });
    }

    /// spawn a new slow job on a certain NAME IF it can run immediately
    #[allow(clippy::result_unit_err)]
    pub fn try_run<F>(&self, name: &Name, f: F) -> Result<SlowJob, ()>
    where
        F: /*FnOnce()*/Future<Output=()> + Send + 'static,
    {
        let mut lock = self.internal.lock().expect("lock poisoned while try_run");
        let lock = &mut *lock;
        //spawn already queued
        lock.spawn_queued(self);
        if lock.can_spawn(name) {
            Ok(lock.spawn(self, true, name, f))
        } else {
            Err(())
        }
    }

    pub fn spawn<F>(&self, name: &Name, f: F) -> SlowJob
    where
        F: /*FnOnce()*/Future<Output=()> + Send + 'static,
    {
        self.internal
            .lock()
            .expect("lock poisoned while spawn")
            .spawn(self, true, name, f)
    }

    /// Spawn at the front of the queue, which is preferrable in some cases.
    pub fn spawn_front<F>(&self, name: &Name, f: F) -> SlowJob
    where
        F: /*FnOnce()*/Future<Output=()> + Send + 'static,
    {
        self.internal
            .lock()
            .expect("lock poisoned while spawn")
            .spawn(self, false, name, f)
    }

    pub fn take_metrics(&self) -> HashMap<Name, Vec<JobMetrics>> {
        self.internal
            .lock()
            .expect("lock poisoned while take_metrics")
            .take_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_pool(
        pool_threads: usize,
        global_threads: u64,
        metrics: usize,
        foo: u64,
        bar: u64,
        baz: u64,
    ) -> SlowJobPool {
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .unwrap();
        let pool = SlowJobPool::new(global_threads, metrics, threadpool);
        if foo != 0 {
            pool.configure("FOO", |x| x / foo);
        }
        if bar != 0 {
            pool.configure("BAR", |x| x / bar);
        }
        if baz != 0 {
            pool.configure("BAZ", |x| x / baz);
        }
        pool
    }

    #[test]
    fn simple_queue() {
        let pool = mock_pool(4, 4, 0, 1, 0, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 1u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 4);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "FOO");
    }

    #[test]
    fn multiple_queue() {
        let pool = mock_pool(4, 4, 0, 1, 0, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 2u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 4);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "FOO");
        assert_eq!(result[1], "FOO");
    }

    #[test]
    fn limit_queue() {
        let pool = mock_pool(5, 5, 0, 1, 0, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 80u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "FOO");
        assert_eq!(result[1], "FOO");
        assert_eq!(result[2], "FOO");
        assert_eq!(result[3], "FOO");
    }

    #[test]
    fn simple_queue_2() {
        let pool = mock_pool(4, 4, 0, 1, 1, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 1u64), ("BAR", 1u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 4);
        assert_eq!(result.len(), 2);
        assert_eq!(result.iter().filter(|&x| x == "FOO").count(), 1);
        assert_eq!(result.iter().filter(|&x| x == "BAR").count(), 1);
    }

    #[test]
    fn multiple_queue_3() {
        let pool = mock_pool(4, 4, 0, 1, 1, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 2u64), ("BAR", 2u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result.iter().filter(|&x| x == "FOO").count(), 2);
        assert_eq!(result.iter().filter(|&x| x == "BAR").count(), 2);
    }

    #[test]
    fn multiple_queue_4() {
        let pool = mock_pool(4, 4, 0, 2, 1, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 3u64), ("BAR", 3u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result.iter().filter(|&x| x == "FOO").count(), 2);
        assert_eq!(result.iter().filter(|&x| x == "BAR").count(), 2);
    }

    #[test]
    fn multiple_queue_5() {
        let pool = mock_pool(4, 4, 0, 2, 1, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 5u64), ("BAR", 5u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 5);
        assert_eq!(result.len(), 5);
        assert_eq!(result.iter().filter(|&x| x == "FOO").count(), 2);
        assert_eq!(result.iter().filter(|&x| x == "BAR").count(), 3);
    }

    #[test]
    fn multiple_queue_6() {
        let pool = mock_pool(40, 40, 0, 2, 1, 0);
        let internal = pool.internal.lock().unwrap();
        let queue_data = [("FOO", 5u64), ("BAR", 5u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        let result = internal.calc_queued_order(queued, 11);
        assert_eq!(result.len(), 10);
        assert_eq!(result.iter().filter(|&x| x == "FOO").count(), 5);
        assert_eq!(result.iter().filter(|&x| x == "BAR").count(), 5);
    }

    #[test]
    fn roundrobin() {
        let pool = mock_pool(4, 4, 0, 2, 2, 0);
        let queue_data = [("FOO", 5u64), ("BAR", 5u64)]
            .iter()
            .map(|(n, c)| ((*n).to_owned(), *c))
            .collect::<Vec<_>>();
        let queued = queue_data
            .iter()
            .map(|(s, c)| (s, *c))
            .collect::<HashMap<_, _>>();
        // Spawn a FOO task.
        pool.internal
            .lock()
            .unwrap()
            .spawn("FOO", || println!("foo"));
        // a barrier in f doesnt work as we need to wait for the cleanup
        while pool.internal.lock().unwrap().global_spawned_and_running != 0 {
            std::thread::yield_now();
        }
        let result = pool
            .internal
            .lock()
            .unwrap()
            .calc_queued_order(queued.clone(), 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "BAR");
        // keep order if no new is spawned
        let result = pool
            .internal
            .lock()
            .unwrap()
            .calc_queued_order(queued.clone(), 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "BAR");
        // spawn a BAR task
        pool.internal
            .lock()
            .unwrap()
            .spawn("BAR", || println!("bar"));
        while pool.internal.lock().unwrap().global_spawned_and_running != 0 {
            std::thread::yield_now();
        }
        let result = pool.internal.lock().unwrap().calc_queued_order(queued, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "FOO");
    }

    #[test]
    #[should_panic]
    fn unconfigured() {
        let pool = mock_pool(4, 4, 0, 2, 1, 0);
        let mut internal = pool.internal.lock().unwrap();
        internal.spawn("UNCONFIGURED", || println!());
    }

    #[test]
    fn correct_spawn_doesnt_panic() {
        let pool = mock_pool(4, 4, 0, 2, 1, 0);
        let mut internal = pool.internal.lock().unwrap();
        internal.spawn("FOO", || println!("foo"));
        internal.spawn("BAR", || println!("bar"));
    }

    #[test]
    fn can_spawn() {
        let pool = mock_pool(4, 4, 0, 2, 1, 0);
        let internal = pool.internal.lock().unwrap();
        assert!(internal.can_spawn("FOO"));
        assert!(internal.can_spawn("BAR"));
    }

    #[test]
    fn try_run_works() {
        let pool = mock_pool(4, 4, 0, 2, 1, 0);
        pool.try_run("FOO", || println!("foo")).unwrap();
        pool.try_run("BAR", || println!("bar")).unwrap();
    }

    #[test]
    fn try_run_exhausted() {
        use std::{thread::sleep, time::Duration};
        let pool = mock_pool(8, 8, 0, 4, 2, 0);
        let func = || loop {
            sleep(Duration::from_secs(1))
        };
        pool.try_run("FOO", func).unwrap();
        pool.try_run("BAR", func).unwrap();
        pool.try_run("FOO", func).unwrap();
        pool.try_run("BAR", func).unwrap();
        pool.try_run("FOO", func).unwrap_err();
        pool.try_run("BAR", func).unwrap();
        pool.try_run("FOO", func).unwrap_err();
        pool.try_run("BAR", func).unwrap();
        pool.try_run("FOO", func).unwrap_err();
        pool.try_run("BAR", func).unwrap_err();
        pool.try_run("FOO", func).unwrap_err();
    }

    #[test]
    fn actually_runs_1() {
        let pool = mock_pool(4, 4, 0, 0, 0, 1);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);
        pool.try_run("BAZ", move || {
            barrier_clone.wait();
        })
        .unwrap();
        barrier.wait();
    }

    #[test]
    fn actually_runs_2() {
        let pool = mock_pool(4, 4, 0, 0, 0, 1);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);
        pool.spawn("BAZ", move || {
            barrier_clone.wait();
        });
        barrier.wait();
    }

    #[test]
    fn actually_waits() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Barrier,
        };
        let pool = mock_pool(4, 4, 0, 4, 0, 1);
        let ops_i_ran = Arc::new(AtomicBool::new(false));
        let ops_i_ran_clone = Arc::clone(&ops_i_ran);
        let barrier = Arc::new(Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);
        let barrier2 = Arc::new(Barrier::new(2));
        let barrier2_clone = Arc::clone(&barrier2);
        pool.try_run("FOO", move || {
            barrier_clone.wait();
        })
        .unwrap();
        pool.spawn("FOO", move || {
            ops_i_ran_clone.store(true, Ordering::SeqCst);
            barrier2_clone.wait();
        });
        // in this case we have to sleep
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(!ops_i_ran.load(Ordering::SeqCst));
        // now finish the first job
        barrier.wait();
        // now wait on the second job to be actually finished
        barrier2.wait();
    }

    #[test]
    fn verify_metrics() {
        use std::sync::Barrier;
        let pool = mock_pool(4, 4, 2, 1, 0, 4);
        let barrier = Arc::new(Barrier::new(5));
        for name in &["FOO", "BAZ", "FOO", "FOO"] {
            let barrier_clone = Arc::clone(&barrier);
            pool.spawn(name, move || {
                barrier_clone.wait();
            });
        }
        // now finish all jobs
        barrier.wait();
        // in this case we have to sleep to give it some time to store all the metrics
        std::thread::sleep(std::time::Duration::from_secs(2));
        let metrics = pool.take_metrics();
        let foo = metrics.get("FOO").expect("FOO doesn't exist in metrics");
        //its limited to 2, even though we had 3 jobs
        assert_eq!(foo.len(), 2);
        assert!(metrics.get("BAR").is_none());
        let baz = metrics.get("BAZ").expect("BAZ doesn't exist in metrics");
        assert_eq!(baz.len(), 1);
    }
}
