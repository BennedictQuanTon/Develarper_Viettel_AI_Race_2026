# Image repo/tag for Hub (override on CLI):
#   make push IMAGE_REPO=myuser/develarper-lfm25 TAG=p0

IMAGE_REPO ?= YOUR_DOCKERHUB/develarper-lfm25
TAG        ?= p0
PLATFORM   ?= linux/amd64
VLLM_IMAGE ?= vllm/vllm-openai:v0.23.0

.PHONY: help download-model preflight build build-baseline push tag-digest ers-sim smoke-mock submit-p0-compose

help:
	@echo "Targets:"
	@echo "  make download-model   # HF -> model_weights/LFM2.5-1.2B-Instruct"
	@echo "  make preflight        # local checks before push"
	@echo "  make build            # docker buildx linux/amd64 with baked /model"
	@echo "  make push             # push $(IMAGE_REPO):$(TAG)"
	@echo "  make ers-sim          # synthetic ERS ranking signal"
	@echo "  make smoke-mock       # CPU OpenAI smoke"
	@echo ""
	@echo "Override: IMAGE_REPO=you/develarper-lfm25 TAG=p0 VLLM_IMAGE=vllm/vllm-openai:v0.23.0"

download-model:
	bash scripts/download_model.sh

preflight:
	bash scripts/preflight.sh

ers-sim:
	python3 scripts/ers_sim.py --meta eval/traces/example_meta.json --params configs/ers_params.example.json --compare-baseline

smoke-mock:
	python3 scripts/mock_openai_server.py --port 8000

build:
	@test -f model_weights/LFM2.5-1.2B-Instruct/config.json || (echo "Run make download-model first"; exit 1)
	docker buildx build \
	  --platform $(PLATFORM) \
	  --build-arg VLLM_IMAGE=$(VLLM_IMAGE) \
	  -t $(IMAGE_REPO):$(TAG) \
	  --load \
	  .

# Optional: try BTC baseline tag (may fail LFM2.5 load — verify before using)
build-baseline:
	@test -f model_weights/LFM2.5-1.2B-Instruct/config.json || (echo "Run make download-model first"; exit 1)
	docker buildx build \
	  --platform $(PLATFORM) \
	  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.22.1 \
	  -t $(IMAGE_REPO):baseline-0221 \
	  --load \
	  .

push:
	docker push $(IMAGE_REPO):$(TAG)
	@echo "Pinned digest:"
	@docker image inspect $(IMAGE_REPO):$(TAG) --format '{{index .RepoDigests 0}}' || true
	@echo "Update docker-compose.submit.yml image: field, then upload to portal."

submit-p0-compose:
	@cp docker-compose.submit.yml /tmp/docker-compose.yml
	@echo "Wrote /tmp/docker-compose.yml — edit image then upload to portal"

tag-digest:
	docker image inspect $(IMAGE_REPO):$(TAG) --format '{{index .RepoDigests 0}}'
