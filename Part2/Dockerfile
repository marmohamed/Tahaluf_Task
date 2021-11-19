FROM conda/miniconda3
WORKDIR /app
COPY ./environment.yml /app/environment.yml
RUN conda env create --file /app/environment.yml --name task_2
COPY . /app
RUN /bin/bash -c "source activate task_2"
# RUN groupadd -r task && useradd -g task mariam
# RUN chown -R task:mariam /app
# USER mariam
CMD bash scripts/train.sh