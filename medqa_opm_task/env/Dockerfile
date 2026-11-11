FROM python:3.11-slim

WORKDIR /task
COPY env/requirements.txt /tmp/task-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/task-requirements.txt
COPY . /task

CMD ["/bin/bash"]
