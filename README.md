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
##### Countires used
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
* Ukraine
* US
#### Stage A
##### Features used
For features we used the following
* function word's term frequency
* 1000 most common words
* 1000 most common part of speech trigrams
* 1000 most common char trigrams
##### Classifier
The features were trained using Logistic regression classifier
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
| Ukraine | | | |
| US | | | |
| Total | | | |

###### Out-of-Domain
Results per-country are accuracy

Combined Accuracy is at the bottom row

| Country | Is Native | Language Family | Native Language |
| --- | :---: | :---: | :---: |
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
| Ukraine | | | |
| US | | | |
| Total | | | |
