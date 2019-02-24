FROM tensorflow/tensorflow

WORKDIR /workdir

RUN pip install tensorflowjs

ENTRYPOINT [ "tensorflowjs_converter" ]

CMD ["--help"]
