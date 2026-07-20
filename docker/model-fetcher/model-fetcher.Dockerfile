FROM python:3.10-slim

RUN python -m pip install --no-cache-dir dvc

WORKDIR /repo

COPY .dvc/ /repo/.dvc/
COPY .dvcignore /repo/.dvcignore
COPY .git/ /repo/.git/
COPY models/ /repo/models_source/

COPY docker/model-fetcher/fetch_model.sh /repo/fetch_model.sh
RUN sed -i 's/\r$//' /repo/fetch_model.sh && chmod +x /repo/fetch_model.sh

CMD ["/repo/fetch_model.sh"]
