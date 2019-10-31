FROM nvidia/cuda:10.0-cudnn7-runtime
MAINTAINER tanaka504 <nishikigi.nlp@gmail.com>

# apt-get
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install git vim curl locales mecab libmecab-dev mecab-ipadic-utf8 make xz-utils file sudo bzip2 wget python3-pip swig
RUN apt-get -y install libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev zlib1g-dev

# install pyenv
ENV HOME /home
RUN git clone https://github.com/yyuu/pyenv.git $HOME/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    eval "$(pyenv init -)"

RUN pyenv install 3.7.5
RUN pyenv global 3.7.5

# install python3 packages
RUN pip3 install --upgrade pip
RUN pip3 install mecab-python3

# character encoding
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# add MeCab Neologd
WORKDIR /usr/src/
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /usr/src/mecab-ipadic-neologd && \
/usr/src/mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y
RUN mecab -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/

WORKDIR /home/
RUN git clone https://github.com/tanaka504/RLResponseGeneration.git
CMD ["/bin/bash"]
