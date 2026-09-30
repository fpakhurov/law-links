IMAGE ?= law-links-service
CONTAINER ?= law-links-container

.PHONY: install test eval speed noise audit train-chains run docker-build docker-run docker-stop smoke

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

eval:
	python -m scripts.eval --verbose

speed:
	python -m scripts.speed

noise:
	python -m research.noise

audit:
	python -m scripts.audit_aliases

train-chains:
	python -m research.train_chains

run:
	uvicorn main:app --host 0.0.0.0 --port 8978 --reload

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run -d -p 8978:8978 --name $(CONTAINER) $(IMAGE)

docker-stop:
	docker stop $(CONTAINER) && docker rm $(CONTAINER)

smoke:
	curl -sf -X POST "http://localhost:8978/detect" \
		-H "Content-Type: application/json" \
		-d '{"text": "Согласно статье 23 Налогового кодекса и ст. 145 Гражданского кодекса"}'
