## Ejercicio 6

#### Dependencias

- Instalar **python**
- Ejecute sobre un terminal `pip install numpy langdetect`

#### Ejecución (6.a,  6.b)

En una terminal correr el siguiente comando:

```bash
python langdectector.py -h
```
El cual desplegará el `help` del script:

```bash
usage: langdectector.py [-h] [-t TRAIN] [-p PREDICT] [-r RESULTS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        files or directory to train models, default: None
  -p PREDICT, --predict PREDICT
                        files or directory to predict language, default: None
  -r RESULTS, --results RESULTS

                        files or directory that contains the expected result,
                        default: None
  -v, --verbose         show messages during process
```

#### Ejemplos

Para ejecutar lo solicitado en el **Ejercicio 6**, pegue el siguiente comando en una terminal sobre el siguiente directorio `TP - 02/6/a` ó `TP - 02/6/b`

```bash
./langdectector.py -t ./languageIdentificationData/training -p ../languageIdentificationData/test -r ../languageIdentificationData/solution
```

**Nota:** Para forzar el que el script realize el training eliminar la carpeta `/.tran-data`

#### Ejecución (6.c)

En una terminal correr el siguiente comando:

```bash
python langdectector.py -h
```
El cual desplegará el `help` del script:

```bash
usage: langdectector.py [-h] [-p PREDICT] [-r RESULTS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -p PREDICT, --predict PREDICT
                        files or directory to predict language, default: None
  -r RESULTS, --results RESULTS
                        files or directory that contains the expected result,
                        default: None
  -v, --verbose         show messages during process
```

#### Ejemplos

Para ejecutar lo solicitado en el **Ejercicio 6**, pegue el siguiente comando en una terminal sobre el siguiente directorio `TP - 02/6/c`

```bash
./langdectector.py -p ../languageIdentificationData/test -r ..languageIdentificationData/solution
```
