# Docker images for Python

Supporting material for blogpost [agramunt.me/posts/python-docker/](https://agramunt.me/posts/python-docker/).

## Instructions

In `build-run.sh` uncomment the image you want to build and run then execute

```bash
export TASK=build_image
./build-run.sh

export TASK=run_image
./build-run.sh
```

After this you will be ssh'd into your new container.