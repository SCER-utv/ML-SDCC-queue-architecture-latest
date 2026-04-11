
 
# Modificare codice e mettere up su gh e docker  #

clonati repo da github
Modifichi codice

## Impacchetta facendo nel terminale della cartella del progetto:

``` docker build --platform linux/amd64 -t emarusso/distributed-rf:latest . ```

## loggati al tuo account, sempre nello stesso terminale facendo:

``` docker login ```

## spedisci online l'immagine

``` docker push emarusso/distributed-rf:latest ```


## SALVA SUBITO SU GITHUB, così è consistente###

# Runnare codice 
## master ##
Avvia istanza master su ec2 instances
connettiti ssh

### Se istanza nuova ###

``` 
sudo apt-get update -y
```
``` 
sudo apt-get install docker.io -y
``` 

``` 
sudo systemctl start docker
``` 
``` 
sudo systemctl enable docker
``` 

``` 
sudo docker pull emarusso/distributed-rf:latest
```
``` 
sudo docker run -d \ --name master-node \ --restart always \ emarusso/distributed-rf:latest \ python -u src/master.py
```


### Se istanza l'avevi già avviata in passato ti basta fare:
 
 ``` 
 sudo docker pull emarusso/distributed-rf:latest
``` 
``` 
sudo docker run -d \ --name master-node \ --restart always \ emarusso/distributed-rf:latest \ python -u src/master.py
```


### visualizza log master

``` 
sudo docker logs -f master-node
```

### per terminare container

``` 
sudo docker rm -f master-node
```



## worker ##

L'istanza si avvierà automaticamente con ASG
connettiti ssh facendo:
 
``` 
ssh -i "distributed-random-forest-key.pem" ec2-user@ip pubblico
```

attendi che docker sia up, lo puoi controllare con:

``` 
sudo docker ps
```

Una volta online, ti basta fare:

``` 
sudo docker logs -f worker-node
```

Per terminare container:

``` 
sudo docker rm -f worker-node
```

# Messaggio del client al master  #
inserisci nella coda sqs un messaggio formattato cosi:

```
{
"dataset": "higgs",
"num_workers":5,
"num_trees":5
}
```




