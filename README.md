# Workflow

## Grab the data somewhere

```
,city1,city2,country1,country2,continent1,continent2,isp1,isp2,target
0,Périgueux,Tangier,FR,MO,EU,AF,Free SAS,Maroc Telecom,618.0291885766151
1,Périgueux,Marseille,FR,FR,EU,EU,Free SAS,Wifirst S.A.S.,711.2331863098909
2,Périgueux,Santa Monica,FR,US,EU,NA,Free SAS,Spectrum,286.6715596960389
3,Périgueux,The Hague,FR,NL,EU,EU,Free SAS,Xs4all Internet BV,751.8453477595747
4,Périgueux,Casablanca,FR,MO,EU,AF,Free SAS,Maroc Telecom,700.8397596354928
5,Périgueux,Doha,FR,QA,EU,AS,Free SAS,Gulf Bridge International Inc.,349.30800668014916
6,Périgueux,Reading,FR,UK,EU,EU,Free SAS,Virgin Media,518.1104757480963
7,Périgueux,Birkeland,FR,NO,EU,EU,Free SAS,Telenor Norge,386.21754911927883
8,Périgueux,Tangier,FR,MO,EU,AF,Free SAS,Maroc Telecom,817.5680478593774
```

Above is a sample of the dataframe that the current pipeline + model accept.

## Training

A tiny cmd tool is written for the task.

### Train the pipeline

Simply type: `pipenv run flask data train-pipe`

### Train the model

Simply type `pipenv run flask data train-model --epoch 30`

## Serving

The image needs to be built before serving,  training pipeline under `pipeline/pipe.joblib` (ml-production) and `model/00001` (tf-model) will be taken into their own image. 

`skaffold run --default-repo eu.gcr.io/easybroadcast-1002` will deploy the service. Please refer to `kubernetes/deployment.yaml`
