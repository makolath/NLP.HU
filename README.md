# NLP HU - Native Language Identification
## Natural Language Processing Lab - Haifa University - Native Language Identification
### Pre-trained models
trained models are available [here](https://drive.google.com/open?id=1PFvy3NKD0Nc3V1LV1G_HZhWg3vCUZTCZ)
### Pre-computed vocabularies
vocabularies are available [here](https://drive.google.com/open?id=1Dq1HrPnJX1LvXuCc-3FjV5uGRInkpHwO)
### Raw data
raw data used for classification is available [here](https://drive.google.com/drive/folders/125RAHvCIHBR-jAUnIhqzWhdxh0mQ_fcv)
### Processed data
Processed data (pre-computed features and their targets) are available [here](https://drive.google.com/open?id=1_wJEDkDpRTX9_EOwHJrISUH4rzp5qLlX)

### Results
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
#### Stage A
##### Features used
For features we used the following
* function word's term frequency
* 1000 most common words
* 1000 most common part of speech trigrams
* 1000 most common char trigrams
##### Classifier
The model was trained using Logistic Regression classifier
##### Results
###### In-Domain (data from r/europe)
Results for all countries are ten-fold cross validation

Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language |
| --- | :---: | :---: | :---: |
| All | 81.9% |56.26% | 48.71% |
| Albania | | | |
| Australia | | | |
| Austria | | | |
| Bosnia | | | |
| Bulgaria | | | |
| Croatia | | | |
| Czech Republic | | | |
| Denmark | | | |
| Estonia | | | |
| Finland | | | |
| France | | | |
| Germany | | | |
| Greece | | | |
| Hungary | | | |
| Iceland | | | |
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
| Total | | | |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language |
| --- | :---: | :---: | :---: |
| Albania | 92.76% | 4.47% | 4.46% |
| Australia | 23.07% | 43.64% | 66.69% |
| Austria | 91.6% | 46.88% | 32.25%|
| Bosnia | 92.83% | 44.14% | 8.41% |
| Bulgaria | 90.27% | 28.97% | 2.62% |
| Croatia | 90.46% | 35.91% | 3.61% |
| Czech Republic | 90.64% | 37.71% | 4.57% |
| Denmark | 93.37% | 52.23% | 32.17% |
| Estonia | 90.22% | 33.24% | 5.04% |
| Finland | 90.17% | 8.86% | 11.45% |
| France | 90.44% | 28.72% | 15.47% |
| Germany | 92.00% | 50.05% | 35.95% |
| Greece | 90.10% | 8.66% | 6.79% |
| Hungary | 91.06% | 5.05% | 3.35% |
| Iceland | 91.05% | 47.05% | 2.13% |
| Ireland | 30.65% | 50.84% | 70.85% |
| Italy | 89.57% | 26.29% | 7.00% |
| Latvia | 95.70% | 52.02% | 10.39% |
| Lithuania | 91.72% | 39.15% | 3.56% |
| Netherlands | 89.14% | 45.03% | 14.41% |
| Norway | 86.64% | 42.21% | 3.68% |
| Poland | 90.80% | 42.63% | 21.74% |
| Portugal | 90.86% | 25.79% | 7.47% |
| Romania | 91.92% | 22.73% | 9.52% |
| Russia | 89.54% | 40.04% | 6.93% |
| Serbia | 89.77% | 35.94% | 2.50% |
| Slovakia | 94.86% | 51.23% | 16.45% |
| Slovenia | 92.10% | 32.17% | 2.22% |
| Spain | 86.93% | 24.68% | 10.11% |
| Sweden | 88.76% | 42.03% | 7.39% |
| Turkey | 89.90% | 15.53% | 13.94% |
| UK | 32.35% | 53.10% | 73.15% |
| US | 21.01% | 42.34% | 65.67% |
| Total | | | |
