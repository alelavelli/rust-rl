format:
	cargo fmt
lint:
	cargo clippy

test:
	cargo test -- --test-threads=6

all: format lint test

doc: 
	cargo doc
	RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps --open -p rlalgs -p rlenv