FROM tensorflow/syntaxnet
MAINTAINER Shubham Bhardwaj <shubham.bhardwaj4245@gmail.com>

RUN apt-get update
RUN apt-get install -y ca-certificates gettext-base vim

COPY ./ /document-embeddings
RUN pip install -r /document-embeddings/requirements.txt
RUN apt-get install blt libtcl8.6 libtk8.6 libxss1 tk8.6-blt2.5
RUN apt-get install python-tk
RUN python -m nltk.downloader punkt

COPY ./models /models
COPY custom_parse.sh /opt/tensorflow/models/syntaxnet/syntaxnet/
RUN chmod 777 /opt/tensorflow/models/syntaxnet/syntaxnet/custom_parse.sh

RUN mkdir data
RUN wget http://www.cs.cornell.edu/~schnabts/eval/wiki_2008.zip -P ./data
RUN unzip data/wiki_2008.zip
RUN rm data/wiki_2008.zip

VOLUME ["/document-embeddings/config","/models"]
WORKDIR /document-embeddings
RUN export LANG=C.UTF-8
ENV PYTHONPATH $PYTHONPATH:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/protobuf/python:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/__main__:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/six_archive:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/org_tensorflow:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/protobuf

