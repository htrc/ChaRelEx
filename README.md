# ChaRelEx

Character Relation Extraction framework for relationship sentiment dynamic analysis on fictional novel.

1. Download the Gutenberg data file using "preparation_gutenberg_download.ipynb" , sample files have been downloaded in the samples folder: "Gutenberg_Volumes_200.json"
2. Run "preparation_parsing_etext.py" to extract information from the text usin Named Entity Recognition (NER) and OpenIE , sample parsed text is available in the samples folder: "gutenberg_parse_ie_etexts.json"
3. Run aligner script "charelex_allignment.py" or "charelex_allignment.ipynb" to prepare the network data and sentiment analysis over the network dynamic, sample network file is available in the samples folder: "gutenberg_network_data.csv"

