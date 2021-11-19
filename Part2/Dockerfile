FROM conda/miniconda3
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN conda env create -f /app/requirements.txt --name task
ADD . /app
RUN /bin/bash -c "source activate task"
RUN groupadd -r task && useradd -g task mariam
RUN chown -R task:mariam /app
USER mariam
CMD python Main.py