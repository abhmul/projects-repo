sudo docker build . -t generalization:latest
sudo docker run --gpus all -v $HOME:$HOME -u $(id -u):$(id -g) -w $HOME/dev/projects-repo/pyprojects/generalization-experiments -e KERAS_HOME=$HOME/.keras -it generalization:latest /bin/bash
