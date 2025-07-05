from transformers import pipeline, set_seed

# Load GPT-2 model
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Take input
keywords = input("Enter keywords separated by commas: ").strip().split(',')
keywords = [word.strip() for word in keywords if word.strip()]

# Create a strong storytelling prompt
prompt = (
    f"Write a creative story of at least 300 words that naturally includes the following words: "
    f"{', '.join(keywords)}. Begin the story like this:\n\n"
    f"Once upon a time, in a quiet village surrounded by hills and forests, "
    f"there lived a curious child who always dreamed of adventure. "
    f"One day, everything changed...\n\n"
)

# Generate story
story_parts = generator(
    prompt,
    max_length=700,
    num_return_sequences=1,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    pad_token_id=50256
)

story = story_parts[0]['generated_text']

# Clean and format
print("\n--- Your Story ---\n")
print(story.strip().replace(". ", ".\n"))

# Save to file
with open("generated_story.txt", "w", encoding="utf-8") as f:
    f.write(story)
    