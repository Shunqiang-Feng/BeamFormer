FROM huggingface/accelerate:gpu-nightly

WORKDIR /app
COPY . .

RUN conda run -n accelerate pip install --no-cache-dir -r requirements.txt

RUN chmod +x run_reproduce_figures.sh

CMD ["conda", "run", "--no-capture-output", "-n", "accelerate", "python", "-m", "beamformer.cdf_plot"]