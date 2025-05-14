# Instructions

Install opencv

```bash
./external/install-opencv.sh
```

Download image

```bash
./scripts/download-img.sh
```

Finally compile and run


```bash
TASK=compile_exec ./scripts/compile-run.sh

./build/bin/main img/raw_img.jpeg
```