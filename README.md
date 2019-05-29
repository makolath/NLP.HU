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
##### Countries used
The following countries data was used
* Albania
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
* Iceland
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

Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| All | 81.9% |56.26% | 48.71% | - |
| Albania | 96.65% | 12.78% | 19.03% | 3920 |
| Australia | 28.53% | 49.75% | 68.39% | 4930 |
| Austria | 92.44% | 57.33% | 48.57% | 14150 |
| Bosnia | 97.19% | 61.72% | 33.90% | 10410 |
| Bulgaria | 92.70% | 36.53% | 14.32% | 10670 |
| Croatia | 91.84% | 47.09% | 12.98% | 10570 |
| Czech Republic | 94.10% | 47.44% | 17.75% | 15200 |
| Denmark | 94.20% | 66.20% | 49.43% | 35520 |
| Estonia | 94.61% | 53.26% | 25.81% | 15850 |
| Finland | 91.73% | 18.70% | 24.90% | 27670 |
| France | 88.22% | 38.13% | 25.27% | 38140 |
| Germany | 92.08% | 59.70% | 50.46% | 85150 |
| Greece | 93.37% | 21.89% | 18.17% | 20300 |
| Hungary | 94.93% | 11.43% | 13.50% | 10160 |
| Iceland | 91.31% | 58.13% | 7.23% | 3800 |
| Ireland | 44.71% | 62.43% | 75.67% | 27960 |
| Italy | 91.31% | 36.63% | 15.38% | 14700 |
| Latvia | 96.61% | 62.97% | 31.07% | 12690 |
| Lithuania | 95.10% | 60.21% | 18.85% | 10400 |
| Netherlands | 88.37% | 53.13% | 24.51% | 52400 |
| Norway | 88.21% | 52.06% | 9.98% | 10560 |
| Poland | 95.34% | 57.98% | 40.02% | 38890 |
| Portugal | 91.03% | 38.48% | 19.67% | 17470 | 
| Romania | 91.06% | 34.48% | 27.47% | 32710 |
| Russia | 94.17% | 59.29% | 19.72% | 13170 |
| Serbia | 92.58% | 46.99% | 8.97% | 7580 |
| Slovakia | 96.54% | 63.96% | 36.30% | 12560 |
| Slovenia | 92.41% | 42.04% | 9.80% | 8690 |
| Spain | 91.80% | 47.47% | 31.43% | 22260 |
| Sweden | 90.59% | 53.25% | 19.49% | 26860 |
| Turkey | 94.63% | 31.04% | 28.93% | 13910 |
| UK | 45.31% | 63.61% | 76.74% | 101250 |
| US | 30.86% | 50.73% | 67.08% | 64540 |
| Ukraine | 97.61% | 72.39% | 31.92% | 12890 |
| Total | 79.53% | 51.19% | 40.57% | 807930 |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| Albania | 92.76% | 4.47% | 4.46% | 35630 |
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
| Iceland | 91.05% | 47.05% | 2.13% | 136420 |
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
| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| All | 75.40% | 24.59% | 0.48% | - |
| Albania | | | | 3920 |
| Australia | | | | 4930 |
| Austria | | | | 14150 |
| Bosnia | | | | 10410 |
| Bulgaria | | | | 10670 |
| Croatia | | | | 10570 |
| Czech Republic | | | | 15200 |
| Denmark | | | | 35520 |
| Estonia| | | | 15850 |
| Finland | | | | 27670 |
| France | | | | 38140 |
| Germany | | | | 85150 |
| Greece | | | | 20300 |
| Hungary | | | | 10160 |
| Iceland | | | | 3800 |
| Ireland | | | | 27960 |
| Italy | | | | 14700 |
| Latvia | | | | 12690 |
| Lithuania | | | | 10400 |
| Netherlands | | | | 52400 |
| Norway | | | | 10560 |
| Poland | | | | 38890 |
| Portugal | | | | 17470 | 
| Romania | | | | 32710 |
| Russia | | | | 13170 |
| Serbia | | | | 7580 |
| Slovakia | | | | 12560 |
| Slovenia | | | | 8690 |
| Spain | | | | 22260 |
| Sweden | | | | 26860 |
| Turkey | | | | 13910 |
| UK | | | | 101250 |
| US | | | | 64540 |
| Ukraine | | | | 12890 |
| Total | | | | 807930 |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language | Number of Chunks |
| --- | :---: | :---: | :---: | :---: |
| Albania | | | | 35630 |
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
| Iceland | | | | 136420 |
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
