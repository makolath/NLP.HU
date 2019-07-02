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
| Bulgaria | 85.86 | 58.09 | 41.35% |
| Croatia | 85.07 | 59.14 | 30.50 |
| Czech Republic | 87.78 | 59.41 | 38.01 |
| Denmark | 89.95 | 63.50 | 56.22 |
| Estonia| 88.37 | 66.83 | 44.82 |
| Finland | 82.59 | 42.61 | 37.66 |
| France | 77.30 | 47.64 | 35.37 |
| Germany | 83.54 | 53.66 | 48.49 |
| Greece | 85.05 | 49.95 | 41.54 |
| Hungary | 90.28 | 50.10 | 45.86 |
| Ireland | 69.25 | 69.25 | 69.25 |
| Italy | 84.03 | 54.03 | 43.63 |
| Latvia | 94.80 | 83.20 | 57.68 |
| Lithuania | 91.80 | 72.55 | 46.30 |
| Netherlands | 72.94 | 42.04 | 26.22 |
| Norway | 77.24 | 53.04 | 37.60 |
| Poland | 89.33 | 66.30 | 41.21 |
| Portugal | | | | 
| Romania | | | |
| Russia | | | |
| Serbia | | | |
| Slovakia | | | |
| Slovenia | | | |
| Spain | | | |
| Sweden | | | |
| Turkey | | | |
| UK | | | |
| US | | | |
| Ukraine | | | |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| Australia | 23.07% | 43.64% | 66.69% | 421020 |
| Austria | 91.6% | 46.88% | 32.25%| 294280 |
| Bosnia | 92.83% | 44.14% | 8.41% | 71010 |
| Bulgaria | 90.27% | 28.97% | 2.62% | 119990 |
| Croatia | 90.46% | 35.91% | 3.61% | 143900 |
| Czech Republic | 90.64% | 37.71% | 4.57% | 175320 |
| Denmark | 93.37% | 52.23% | 32.17% | 681270 |
| Estonia | 90.22% | 33.24% | 5.04% | 110720 |
| Finland | 90.17% | 8.86% | 11.45% | 575030 |
| France | 90.44% | 28.72% | 15.47% | 560100 |
| Germany | 92.00% | 50.05% | 35.95% | 1587250 |
| Greece | 90.10% | 8.66% | 6.79% | 205540 |
| Hungary | 91.06% | 5.05% | 3.35% | 144520 |
| Ireland | 30.65% | 50.84% | 70.85% | 919200 |
| Italy | 89.57% | 26.29% | 7.00% | 246910 |
| Latvia | 95.70% | 52.02% | 10.39% | 90890 |
| Lithuania | 91.72% | 39.15% | 3.56% | 135670 |
| Netherlands | 89.14% | 45.03% | 14.41% | 1246100 |
| Norway | 86.64% | 42.21% | 3.68% | 413880 |
| Poland | 90.80% | 42.63% | 21.74% | 437470 |
| Portugal | 90.86% | 25.79% | 7.47% | 311950 |
| Romania | 91.92% | 22.73% | 9.52% | 292770 |
| Russia | 89.54% | 40.04% | 6.93% | 162160 |
| Serbia | 89.77% | 35.94% | 2.50% | 105890 |
| Slovakia | 94.86% | 51.23% | 16.45% | 110050 |
| Slovenia | 92.10% | 32.17% | 2.22% | 73020 |
| Spain | 86.93% | 24.68% | 10.11% | 330950 |
| Sweden | 88.76% | 42.03% | 7.39% | 772480 |
| Turkey | 89.90% | 15.53% | 13.94% | 177940 |
| UK | 32.35% | 53.10% | 73.15% | 3418210 |
| US | 21.01% | 42.34% | 65.67% | 5436530 |
| Ukraine | 95.23% | 55.73% | 14.69% | 120050 |
| Total | 57.64% | 42.35% | 43.01% | 20064120 |

All the results below are using pre-trained word2vec model
#### Bert
##### Results
###### In-Domain (data from r/europe)
| Country | Is Native | Language Family | Native Language |
| --- | :---: | :---: | :---: |
| All | 75.40% | 24.59% | 0.48% |
| Australia | | | |
| Austria | | | |
| Bosnia | | | |
| Bulgaria | | | |
| Croatia | | | |
| Czech Republic | | | |
| Denmark | | | |
| Estonia| | | |
| Finland | | | |
| France | | | |
| Germany | | | |
| Greece | | | |
| Hungary | | | |
| Ireland | | | |
| Italy | | | |
| Latvia | | | |
| Lithuania | | | |
| Netherlands | | | |
| Norway | | | |
| Poland | | | |
| Portugal | | | | 
| Romania | | | |
| Russia | | | |
| Serbia | | | |
| Slovakia | | | |
| Slovenia | | | |
| Spain | | | |
| Sweden | | | |
| Turkey | | | |
| UK | | | |
| US | | | |
| Ukraine | | | |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| Australia | | | | 421020 |
| Austria | | | | 294280 |
| Bosnia | | | | 71010 |
| Bulgaria | | | | 119990 |
| Croatia | | | | 143900 |
| Czech Republic | | | | 175320 |
| Denmark | | | | 681270 |
| Estonia | | | | 110720 |
| Finland | | | | 575030 |
| France | | | | 560100 |
| Germany | | | | 1587250 |
| Greece | | | | 205540 |
| Hungary | | | | 144520 |
| Ireland | | | | 919200 |
| Italy | | | | 246910 |
| Latvia | | | | 90890 |
| Lithuania | | | | 135670 |
| Netherlands | | | | 1246100 |
| Norway | | | | 413880 |
| Poland | | | | 437470 |
| Portugal | | | | 311950 |
| Romania | | | | 292770 |
| Russia | | | | 162160 |
| Serbia | | | | 105890 |
| Slovakia | | | | 110050 |
| Slovenia | | | | 73020 |
| Spain | | | | 330950 |
| Sweden | | | | 772480 |
| Turkey | | | | 177940 |
| UK | | | | 3418210 |
| US | | | | 5436530 |
| Ukraine | | | | 120050 |
| Total | | | | 20064120 |
