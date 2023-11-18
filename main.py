from transformers import pipeline

def abstract_summarization_demo(original_text):
    # Load the BART model for abstractive summarization
    summarizer = pipeline("summarization")

    # Generate the summary
    summary = summarizer(original_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Print the original text
    print("Original Text:\n", original_text)

    # Print the summarized text
    print("\nSummary:\n", summary[0]['summary_text'])

# Example usage
text_to_summarize = """
The Illuminati (/ɪˌluːmɪˈnɑːti/; plural of Latin illuminatus, 'enlightened') is a name given to
 several groups, both real and fictitious. Historically, the name usually refers to the Bavarian
  Illuminati, an Enlightenment-era secret society founded on 1 May 1776 in Bavaria, today part of 
  Germany. The society's stated goals were to oppose superstition, obscurantism, religious influence 
  over public life, and abuses of state power. "The order of the day," they wrote in their general 
  statutes, "is to put an end to the machinations of the purveyors of injustice, to control them without 
  dominating them."[1] The Illuminati—along with Freemasonry and other secret societies—were outlawed
   through edict by Charles Theodore, Elector of Bavaria, with the encouragement of the Catholic Church, 
   in 1784, 1785, 1787 and 1790.[2] During subsequent years, the group was generally vilified by
    conservative and religious critics who claimed that
 the Illuminati continued underground and were responsible for the French Revolution.
"""

abstract_summarization_demo(text_to_summarize)
