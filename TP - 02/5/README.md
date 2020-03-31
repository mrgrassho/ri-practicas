## Ejercicio 5

#### Dependencias

- Instalar **python**
- Ejecute sobre un terminal `pip install nltk`

#### Ejecución

En una terminal correr el siguiente comando:

```bash
python tokenizer_cinco.py -h
```
El cual desplegará el `help` del script:

```bash
usage: tokenizer_cinco.py [-h] [-n MIN] [-x MAX] [-s STOPWORDS] [-d DIR] [-v]
                          [-t {lancaster,porter}]

optional arguments:
  -h, --help            show this help message and exit
  -n MIN, --min MIN     minimun length of terms
  -x MAX, --max MAX     maximun length of terms
  -s STOPWORDS, --stopwords STOPWORDS
                        file containing stopwords
  -d DIR, --dir DIR     directory to scan, default: current working dir

  -v, --verbose         show messages during process
  -t {lancaster,porter}, --stemmer {lancaster,porter}
                        chose stemmer
```

#### Ejemplos

Para ejecutar lo solicitado en el **Ejercicio 5**, pegue el siguiente comando en una terminal sobre el siguiente directorio `TP - 02/5`

```bash
python tokenizer_cinco.py -s stopwords_spanish.txt -n 3 --dir ../RI-tknz-data
```

La **información solicitada** se almacena en los archivos `frecuencies.json`,  `stats.json` y `terms.json`.
