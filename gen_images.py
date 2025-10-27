# %pip install --upgrade transformers
import time
# Initialize timing array of hashes
timing_data = []

# Line 0: Import and setup the Diffusers
timing_entry = {"description": "Import and setup the Diffusers"}
timing_entry["before_time"] = time.time()
from diffusers import StableDiffusionPipeline
timing_entry["after_time"] = time.time()
timing_data.append(timing_entry)

# Line 1: Create the pipeline
timing_entry = {"description": "Pipeline creation"}
timing_entry["before_time"] = time.time()
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5") #.to("cuda")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
timing_entry["after_time"] = time.time()
timing_data.append(timing_entry)

# Line 2: Generate image
timing_entry = {"description": "Image generation"}
timing_entry["before_time"] = time.time()
image = pipe("A photo of an astronaut riding a horse on the moon").images[0]
timing_entry["after_time"] = time.time()
timing_data.append(timing_entry)

# Line 3: Save image
timing_entry = {"description": "Image saving"}
timing_entry["before_time"] = time.time()
image.save("astronaut_horse.png")
timing_entry["after_time"] = time.time()
timing_data.append(timing_entry)

# Print timing results
print("\nTiming Results:")
print("-" * 50)
for entry in timing_data:
    time_diff = entry["after_time"] - entry["before_time"]
    print(f"{entry['description']}: {time_diff:.4f} seconds")

total_time = timing_data[-1]["after_time"] - timing_data[0]["before_time"]
print(f"\nTotal execution time: {total_time:.4f} seconds")
