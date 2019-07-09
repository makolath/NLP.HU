# NLP HU - Native Language Identification
## Natural Language Processing Lab - Haifa University - Native Language Identification
### Pre-trained models
trained models are available [here](https://drive.google.com/open?id=1PFvy3NKD0Nc3V1LV1G_HZhWg3vCUZTCZ)
### Pre-computed vocabularies
vocabularies are available [here](https://drive.google.com/open?id=1Dq1HrPnJX1LvXuCc-3FjV5uGRInkpHwO)
### Raw data
raw data used for classification is available [here](https://drive.google.com/drive/folders/125RAHvCIHBR-jAUnIhqzWhdxh0mQ_fcv)

### Results
##### Classifier
All models were trained using Logistic Regression classifier

The classified unit was constructed of 10 consecutive sentences

Number of units per country for training was 4930
##### Countries used
The following countries data was used
* Australia
* Austria
* Bosnia
* Bulgaria
* Croatia
* Czech Republic
* Denmark
* Estonia
* Finland
* France
* Germany
* Greece
* Hungary
* Ireland
* Italy
* Latvia
* Lithuania
* Netherlands
* Norway
* Poland
* Portugal
* Romania
* Russia
* Serbia
* Slovakia
* Slovenia
* Spain
* Sweden
* Turkey
* UK
* US
* Ukraine
#### Baseline
##### Features used
For features we used the following
* function word's term frequency
* 1000 most common words
* 1000 most common part of speech trigrams
* 1000 most common char trigrams

The classified unit was constructed of 10 consecutive sentences
##### Results
###### In-Domain (data from r/europe)
Results for all countries are ten-fold cross validation

| Country | Is Native | Language Family | Native Language |
| --- | :---: | :---: | :---: |
| All | 84.18% | 61.80% | 47.47% |
| Australia | 69.25% | 69.25% | 69.25% |
| Austria | 83.54% | 53.66% | 48.49% |
| Bosnia | 95.65% | 87.05% | 65.27% |
| Bulgaria | 85.86% | 58.09% | 41.35% |
| Croatia | 85.07% | 59.14% | 30.50% |
| Czech Republic | 87.78% | 59.41% | 38.01% |
| Denmark | 89.95% | 63.50% | 56.22% |
| Estonia| 88.37% | 66.83% | 44.82% |
| Finland | 82.59% | 42.61% | 37.665 |
| France | 77.30% | 47.64% | 35.37% |
| Germany | 83.54% | 53.66% | 48.49% |
| Greece | 85.05% | 49.95% | 41.54% |
| Hungary | 90.28% | 50.10% | 45.86% |
| Ireland | 69.25% | 69.25% | 69.25% |
| Italy | 84.03% | 54.03% | 43.63% |
| Latvia | 94.80% | 83.20% | 57.68% |
| Lithuania | 91.80% | 72.55% | 46.30% |
| Netherlands | 72.94% | 42.04% | 26.22% |
| Norway | 77.24% | 53.04% | 37.605% |
| Poland | 89.33% | 66.30% | 41.21% |
| Portugal | 80.52% | 50.75% | 39.10% | 
| Romania | 80.54% | 41.58% | 32.19% |
| Russia | 89.95% | 73.81% | 49.35% |
| Serbia | 87.80% | 63.89% | 38.64% |
| Slovakia | 95.27% | 80.28% | 55.78% |
| Slovenia | 84.74% | 55.78% | 30.91% |
| Spain | 53.34% | 63.89% | 85.74% |
| Sweden | 77.26% | 48.19% | 32.04% |
| Turkey | 92.61% | 70.34% | 63.56% |
| UK | 69.25% | 69.25% | 69.25% |
| US | 69.25% | 69.25% | 69.25% |
| Ukraine | 97.11% | 89.31% | 60.91% |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| Australia | 47.88% | 47.88% | 47.88% | 421020 |
| Austria | 76.70% | 33.16% | 25.32% | 294280 |
| Bosnia | 82.34% | 54.37% | 15.01% | 71010 |
| Bulgaria | 73.94% | 31.90% | 5.75% | 119990 |
| Croatia | 74.79% | 38.99% | 8.77% | 143900 |
| Czech Republic | 75.88% | 37.69% | 8.01% | 175320 |
| Denmark | 80.13% | 33.42% | 21.56% | 681270 |
| Estonia | 73.47% | 34.88% | 7.17% | 110720 |
| Finland | 73.37% | 18.12% | 12.14% | 575030 |
| France | 73.58% | 24.00% | 13.13% | 560100 |
| Germany | 77.44% | 34.09% | 26.01% | 1587250 |
| Greece | 73.65% | 18.10% | 9.14% | 205540 |
| Hungary | 76.29% | 17.25% | 9.86% | 144520 |
| Ireland | 52.17% | 52.17% | 52.17% | 919200 |
| Italy | 74.12% | 24.15% | 13.53% | 246910 |
| Latvia | 86.07% | 58.77% | 17.19% | 90890 |
| Lithuania | 76.59% | 40.99% | 10.33% | 135670 |
| Netherlands | 71.21% | 28.24% | 9.81% | 1246100 |
| Norway | 68.17% | 29.72% | 10.79% | 413880 |
| Poland | 76.15% | 42.31% | 12.33% | 437470 |
| Portugal | 75.18% | 22.66% | 10.45% | 311950 |
| Romania | 76.39% | 18.42% | 6.74% | 292770 |
| Russia | 74.12% | 42.46% | 13.34% | 162160 |
| Serbia | 74.36% | 38.85% | 7.91% | 105890 |
| Slovakia | 86.14% | 59.08% | 26.25% | 110050 |
| Slovenia | 76.87% | 36.48% | 10.72% | 73020 |
| Spain | 69.94% | 22.91% | 11.78% | 330950 |
| Sweden | 70.08% | 27.04% | 8.23% | 772480 |
| Turkey | 76.78% | 28.72% | 21.14% | 177940 |
| UK | 52.70% | 52.70% | 52.70% | 3418210 |
| US | 43.86% | 43.86% | 43.86% | 5436530 |
| Ukraine | 87.49% | 63.79% | 22.07% | 120050 |
| Total | 60.94% | 39.40% | 31.66% | 19892070 |

All the results below are using pre-trained word2vec model
#### Google's word2vec
##### Results
###### In-Domain (data from r/europe)
| Country | Is Native | Language Family | Native Language |
| --- | :---: | :---: | :---: |
| All | 77.60% | 48.61% | 29.14% |
| Australia | 66.35% | 66.35% | 66.35% |
| Austria | 75.59% | 39.79% | 34.36% |
| Bosnia | 88.09% | 70.95% | 41.36% |
| Bulgaria | 77.24% | 39.55% | 10.81% |
| Croatia | 78.72% | 44.83% | 8.03% |
| Czech Republic | 79.45% | 44.71% | 15.03% |
| Denmark | 78.88% | 42.19% | 25.27% |
| Estonia | 80.65% | 51.44% | 15.84% |
| Finland | 74.38% | 18.05% | 12.66% |
| France | 71.70% | 39.07% | 28.68% |
| Germany | 75.59% | 39.79% | 34.36% |
| Greece | 77.69% | 38.60% | 28.48% |
| Hungary | 80.67% | 23.10% | 17.16% |
| Ireland | 66.35% | 66.35% | 66.35% |
| Italy | 79.35% | 43.73% | 33.63% |
| Latvia | 86.86% | 65.05% | 22.11% |
| Lithuania | 83.35% | 57.81% | 17.24% |
| Netherlands | 66.04% | 29.19% | 9.17% |
| Norway | 69.59% | 35.88% | 13.41% |
| Poland | 81.14% | 53.65% | 22.96% |
| Portugal | 71.68% | 37.44% | 20.85% | 
| Romania | 70.93% | 18.44% | 8.32% |
| Russia | 86.13% | 70.28% | 32.56% |
| Serbia | 83.23% | 53.73% | 24.52% |
| Slovakia | 86.88% | 60.79% | 28.82% |
| Slovenia | 75.86% | 37.34% | 5.80% |
| Spain | 82.68% | 57.12% | 46.69% |
| Sweden | 74.28% | 37.67% | 19.19% |
| Turkey | 88.82% | 60.49% | 54.16% |
| UK | 66.35% | 66.35% | 66.35% |
| US | 66.35% | 66.35% | 66.35% |
| Ukraine | 92.23% | 79.57% | 35.68% |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| Australia | 56.18% | 56.18% | 56.18% | 421020 |
| Austria | 74.91% | 28.94% | 20.61% | 294280 |
| Bosnia | 79.34% | 46.37% | 11.04% | 71010 |
| Bulgaria | 72.28% | 30.96% | 1.30% | 119990 |
| Croatia | 73.53% | 36.26% | 5.20% | 143900 |
| Czech Republic | 73.79% | 34.28% | 1.53% | 175320 |
| Denmark | 73.38% | 26.73% | 10.21% | 681270 |
| Estonia | 69.33% | 32.72% | 1.66% | 110720 |
| Finland | 71.97% | 12.28% | 6.92% | 575030 |
| France | 74.59% | 25.57% | 15.10% | 560100 |
| Germany | 74.40% | 30.10% | 22.40% | 1587250 |
| Greece | 73.37% | 14.82% | 6.32% | 205540 |
| Hungary | 75.10% | 11.60% | 2.46% | 144520 |
| Ireland | 58.25% | 58.25% | 58.25% | 919200 |
| Italy | 75.29% | 27.84% | 15.16% | 246910 |
| Latvia | 80.67% | 45.91% | 5.66% | 90890 |
| Lithuania | 75.85% | 37.32% | 6.11% | 135670 |
| Netherlands | 69.35% | 23.72% | 3.50% | 1246100 |
| Norway | 68.37% | 23.44% | 6.07% | 413880 |
| Poland | 75.19% | 40.08% | 8.49% | 437470 |
| Portugal | 76.65% | 26.34% | 9.86% | 311950 |
| Romania | 75.11% | 16.43% | 1.33% | 292770 |
| Russia | 72.71% | 43.18% | 10.89% | 162160 |
| Serbia | 73.98% | 38.13% | 6.28% | 105890 |
| Slovakia | 81.66% | 46.37% | 12.11% | 110050 |
| Slovenia | 73.96% | 32.79% | 6.98% | 73020 |
| Spain | 70.25% | 23.72% | 15.16% | 330950 |
| Sweden | 69.77% | 24.65% | 5.37% | 772480 |
| Turkey | 79.56% | 29.82% | 25.37% | 177940 |
| UK | 59.74% | 59.74% | 59.74% | 3418210 |
| US | 47.62% | 47.62% | 47.62% | 5436530 |
| Ukraine | 80.46% | 54.64% | 16.27% | 120050 |
| Total | | | | 19892070 |
