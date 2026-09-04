.PHONY: all data features baseline forecast risk dashboard api clean

# Run the entire pipeline end-to-end
all: data features baseline forecast risk
	@echo "Pipeline and model training complete!"

# Step 1: Fetch data and run preprocessing pipeline
data:
	python -m src.data_fetcher
	python -m src.pipeline

# Step 2: Feature engineering
features:
	python -m src.features

# Step 3: Build seasonal-naive baseline
baseline:
	python -m src.baseline

# Step 4: Train forecast models (LightGBM + PyTorch LSTM)
forecast:
	python -m src.forecast

# Step 5: Run risk scoring
risk:
	python -m src.risk

# Launch the Streamlit dashboard
dashboard:
	streamlit run app/streamlit_app.py

# Launch the FastAPI scoring service
api:
	uvicorn service.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Clean generated files
clean:
	rm -rf data/processed/cleaned/*.csv data/processed/features/*.csv data/processed/forecasts/*.csv data/processed/dashboard/*.csv
	rm -rf models/baseline/* models/lightgbm/* models/lstm/*
	rm -f pipeline.log
