# KNN-Cancer-Detection
Modelo em KNN que determina se um indivíduo tem cancêr de pele ou não.

### Sobre o Algoritmo KNN

KNN é a sigla de K-Nearest-Neighbours (ou K vizinhos mais próximos). Trata-se de um algoritmo do tipo de **aprendizado supervisionado** que é muito usado para classificação, mas também pode ser aplicado em problemas de regressão. 

O KNN faz uma clusterização dos dados, separando-os e classificando eles com base na ideia de que **dados semelhantes tendem a estar próximos uns dos outros em um espaço multidimensional**.

Nesse sentido, a ideia principal é encontrar o valor do dado com base nos vizinhos mais próximos dele.

![KNN-Foto](https://miro.medium.com/v2/resize:fit:1400/0*GPaI5OjY4Y8DW4he)

Como foi colocado acima, o **K é o número de vizinhos mais próximos que serão considerados para fazer uma previsão**, e o valor dele pode ser ajustado de acordo com a situação. Temos que tomar cuidado, porque valores pequenos de K podem levar a previsões erradas e sensíveis à outliers, enquanto valores grandes de K pode suavizar demais o resultado.

![KNN-Foto](https://i.imgur.com/eQB5R9j.png)

Para determinar a proximidade entre os pontos dos dados, temos que decidir uma métrica de distância, como **a distância euclidiana, a distância de Manhattan ou outras possíveis métricas**, mas essa escolha depende dos dados e da natureza deles.

Na **classificação**, o KNN calcula a classe mais frequente entre os K vizinhos mais próximos e atribui essa classe ao dado que queremos prever. O KNN pode usar votação ponderada, onde os vizinhos mais próximos tem mais influência.

![KNN-Foto](https://i.imgur.com/fF3gO6F.png)

Na **regressão**, o KNN calcula a **média ou outra medida estatística** dos valores alvo dos K vizinhos mais próximos e atribui esse valor ao novo dado.

![KNN-Foto](https://i.stack.imgur.com/gAILq.png)