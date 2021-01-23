FROM debian:10-slim
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y gcc curl git
COPY . /opt/appbuild/torvus
WORKDIR /opt/appbuild/torvus
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain `cat rust-toolchain`
RUN /bin/bash -c "source $HOME/.cargo/env && cargo build --release --bin veloren-torvus"

FROM debian:10-slim
WORKDIR /opt/app
COPY --from=0 /opt/appbuild/torvus/target/release/veloren-torvus .
# Due to our asset system being crap
COPY ./assets ./assets
CMD ["./veloren-torvus"]
