#!/usr/bin/env bash

# Runs the notebook server.
set -eu
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE_NAME="pytorch/huggingface-transformers"

build() {
  pushd "${SCRIPT_DIR}/docker"
  docker build -t "${IMAGE_NAME}" .
  popd
}


main() {
  echo "Building image"
  build
  echo "Done"

  echo "Running image"
  docker run -it "${IMAGE_NAME}"
}

main "$@"
