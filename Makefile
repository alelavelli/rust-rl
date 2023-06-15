format:
	cargo fmt
lint:
	cargo clippy

test:
	cargo test -- --test-threads=6

all: format lint test