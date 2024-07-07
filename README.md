# cnn_vit_benchmarks

Do not even consider this code close to readable, it's a quick exploration :)

# Run on runpod:

```
> unminimize
> apt install tmux fish vim
> env SHELL=`which fish` tmux
> git clone https://github.com/lucasb-eyer/cnn_vit_benchmarks.git
> cd cnn_vit_benchmarks
> pip3 install -r requirements.txt
> ./untilfail.bash python main.py
> upload.bash
```

For running the local attention benchmark properly apply this hacky patch to timm:

```
> cd /usr/local/python3.10/dist-packages/timm
> git apply ~/cnn_vit_benchmarks/timm_vitdet.patch
```
