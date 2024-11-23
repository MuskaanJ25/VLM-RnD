# Indicness Metric for Visual Language Model Captions (Pipeline Model)

## Introduction

This report details the technical implementation of the Indicness evaluation metric for VLM-generated
captions of images. This metric is realized through a pipeline model, a sequence of actions designed to
evaluate the caption and measure its "Indian-ness". The pipeline integrates multiple computer vision and
natural language processing techniques to analyze the image and its caption, assessing various cultural
and contextual aspects to determine the relevance and representation of Indian elements in the generated
text for an image.

## Overview

This pipeline analyzes both images and their corresponding captions to assess how accurately the captions represent Indian cultural elements present in the images. The system provides a numerical score indicating the caption's effectiveness in capturing culturally significant details.

### Input
- Image file
- Corresponding VLM-generated caption

### Output
- Numerical score representing caption accuracy in cultural context
- Detailed analysis of identified cultural elements (can be shown by printing each component of pipeline results)

## Installation

### Prerequisites

1. Python 3.7+
2. SpaCy English language model:
```bash
python -m spacy download en_core_web_sm
```


### Setup

1. Clone the repository:
```bash
git clone [repository-url]
# OR
# Download and extract the ZIP file from the repository
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Additional dependencies may need to be installed as you encounter them. The system will indicate missing packages through error messages.

## Usage

### Basic Usage

Run the main script:
```python
python main.py
```

### Custom Implementation

The main function accepts image path and caption as arguments:
```python
# ...
if __name__ == "__main__":
   # ...
   score = main(image_path="path/to/image", caption="image caption")
   # ...
```

## Pipeline Architecture

### Components

1. **Image Analysis Module**
   - Processes visual elements
   - Identifies cultural markers accord to each pipeline component

2. **Caption Analysis Module**
   - Natural Language Processing
   - Cultural context evaluation

3. **Correlation Engine**
   - Matches visual elements with textual descriptions
   - Calculates accuracy scores
   - Generates detailed reports

## Technical Details

### Processing Flow

1. Image input processing
2. Caption text analysis
3. Parallel processing of visual and textual elements
4. Correlation analysis
5. Score generation

### Implementation Notes

- Modular architecture for easy maintenance
- Extensible design for adding new cultural elements
- Configurable weight parameters for scoring
- Cache support for YOLO models and NLP toolkits

## Current Features

- Cultural element detection
- Semantic accuracy scoring
- Multi-modal analysis
- Configurable parameters
- Sample test cases included

## Configuration

Weight parameters and other configurations can be adjusted in the config files based on specific requirements.

## Limitations

- Geo-tagging functionality is currently not implemented
- Weight parameters may need further tuning
- Limited to pre-defined cultural symbols (can be extended)

## Future Improvements

- Addition of more cultural symbols
- Enhanced weight parameter optimization
- Extended dataset support
- Geo-tagging implementation


