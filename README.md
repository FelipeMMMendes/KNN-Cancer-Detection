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

Antes de aplicar o KNN, **é essencial preparar os dados**, já que as métricas de distância e escala dos recursos impactam nas previsões do algoritmo.

**Vantagens:**
1. **Simplicidade**: O KNN é um algoritmo simples e fácil de entender.
2. **Adaptabilidade**: O KNN é um algoritmo não paramétrico, o que significa que ele não faz suposições específicas sobre a forma da função que relaciona as entradas às saidas. Isso deixa eke adaptável a uma variedade de problemas, incluindo aqueles em que a relação entre os recursos e as saídas não é linear.
3. **Facilidade de Interpretação**: As previsões do KNN podem ser facilmente interpretadas, podemos identificar quais são os vizinhos mais próximos que contribuem para a classificação de um ponto de dados específico, o que pode ser útil para análise de casos específicos.


**Desvantagens:**
1. **Custo Computacional Alto para Datasets Grandes**: O KNN precisa mapear todas as distâncias entre o ponto de dados desconhecido e todos os pontos de dados no conjunto de treinamento para fazer uma previsão.
2. **Sensível a Características Irrelevantes**: O KNN considera todas as características de maneira igual no cálculo da distância. Se o conjunto de dados tiver características irrelevantes ou ruidosas, elas podem afetar negativamente os resultados do algoritmo.
3. **Necessidade de Ajuste de Hiperparâmetros**: O valor de K é um hiperparâmetro que tem que ser ajustado. Chegar no valor adequado de K pode ser um desafio.
4. **Não lida bem com dados desbalanceados**: Em conjuntos de dados desbalanceados, onde uma classe tem muitos mais exemplos do que outra, o KNN pode ser enviesado em direção à classe majoritária, resultando em classificações menos precisas para a classe minoritária. 

O KNN é aplicado em situações de reconhecimento de padrões, filtragem colaborativa, diagnóstico médico, detecção de fraudes, etc...

O KNN também é chamado de método baseado em distância, porque a lógica principal dele é voltada em torno dos cálculos das distâncias entre pontos de dados.

O KNN pode ser resumido em dois passos principais:
1. **Cálculo de Distâncias**: envolve o cálculo das distâncias entre o ponto de dados desconhecido e todos os outros pontos no conjunto de treinamento. Essas distâncias são calculadas usando métricas como a **distância euclidiana, distância de Manhattan** ou outras métricas personalizadas, dependendo do contexto.
2. **Votação dos K vizinhos mais próximos**: depois do cálculo das distâncias, o KNN considera os valores dos K vizinhos mais próximos do ponto de dados que queremos prever. A classe ou valor mais frequente se torna a previsão final.

#### Normalização dos dados no KNN
Como o KNN depende muito do cálculo das distâncias entre os pontos no espaço de atributos, se os dados não forem normalizados as características com escalas menores podem dominar o cálculo de distâncias, levando a resultados tendenciosos. A normalização garante que todas as características tenham a mesma influência nas distâncias. Além disso, sem a normalização, características com unidades diferentes (tipo quilogramas e metros) podem criar inconsistências nos resultados do KNN. A normalização também melhora o desempenho do KNN, gerando melhores resultados.

#### Min-Max Scaler e Standard Scaler

Duas das técnicas que podem ser usadas para normalização são o **Min-Max Scaler** e o **Standard Scaler**. 

**Min-Max Scaler**: Ele dimensiona as características para um intervalo específico, geralmente entre 0 e 1. É para quando você deseja manter a interpretabilidade das características na mesma escala e quando não quer que uma característica domine as outras devido a valores discrepantes. É muito útil quando os dados têm distribuições não gaussianas.

**Standard Scaler**: Também conhecido como Z-score normalization, ele dimensiona as características de forma que elas tenham média zero e desvio padrão igual a um. É uma escolha para quando queremos remover a média das características e ajustar a variância em torno de 1. É sensível a valores discrepantes, mas isso pode ser benéfico em alguns casos. O Standard Scaler assume que os seus dados estão em uma distribuição gaussiana.

Em resumo, o Min-Max Scaler deve ser usado quando queremos manter as características em uma escala específica e evitar que uma característica domine as outras. O Standard Scaler deve ser usado quando queremos que as características tenham média zero e desvio padrão igual a um, quando os seus dados seguirem uma distribuição gaussiana ou quando os valores discrepantes não forem um problema.

**É importante lembrar que a normalização deve ser aplciada apenas ao conjunto de treinamento, e os mesmos parâmetros de normalização (por exemplo, média e desvio padrão) devem ser usados para normalizar o conjunto de teste ou novos dados.**

#### Distribuição Gaussiana (Normal)

![gaussiana](https://www.inf.ufsc.br/~andre.zibetti/probabilidade/figures/norm1-1.png)


A distribuição gaussiana, também conhecida como distribuição normal, é uma das distribuições de probabilidade mais fundamentais e usadas, ela é caracterizada por:
- **Curva de Sino**: A distribuição gaussiana tem uma forma de curva de sino simétrica em torno de média (ou valor esperado), que é o seu ponto de máximo.
- **Parâmetros**: A distribuição é definida por dois parâmetros principais: a média (μ) e o desvio padrão (σ). A média determina o ponto central da distribuição, enquanto o desvio padrão controla a dispersão dos dados em torno da média.

![media_desvio](https://blog.proffernandamaciel.com.br/wp-content/uploads/2022/07/Captura-de-tela-2022-07-01-195121-768x384.png)

- **Simetria**: A distribuição gaussiana segue uma simetria em relação à sua média, o que significa que metade dos dados está acima da média e a outra metade abaixo dela.

- **Densidade de Probabilidade**: A densidade de probabilidade da distribuição gaussiana mais alta perto da média e diminui à medida que você se afasta dela. Ela é a curva, ela representa a probabilidade de observar valores específicos em uma distribuição.

- **Teorema Central do Limite**: A distribuição gaussiana tem um papel fundamental no **Teorema Central do Limite**, afirma que a soma de N variáveis independentes, com qualquer distribuição e variância semelhantes, é uma variável com distribuição que se aproxima da distribuição normal quando N aumenta.

**Por que não normalizar a variável alvo em uma regressão com KNN**

- **Natureza da Variável Alvo**:
    - A variável alvo em um problema de regressão geralmente representa o valor que você está tentando prever, como preços, pontuações, ou qualquer outra medida numérica contínua.
    - A escala da variável alvo é uma parte intrínseca do problema e representa diretamente a grandeza que você deseja estimar.
- **Métricas de Distância em Atributos (X)**:
    - No KNN para regressão, as métricas de distância são calculadas com base nos atributos (x), não na variável alvo (y).
    - A normalização de y não afeta o cálculo das distâncias entre os pontos de dados com diferentes valores de y.
- **Independência de Escala**:
    - O KNN é um algoritmo que independe das escalas dos valores da variável alvo.
    - A normalização de y não afeta a similaridade ou a distância entre pontos de dados, que é o aspecto crítico do KNN.
- **Interpretação Direta**:
    - A normalização de y pode atrapalhar na interpretação dos resultados do modelo.
    - Manter a variável alvo em sua escala original facilita a compreensão dos resultados, pois eles têm significado direto.
- **Diferentes Alvos, Diferentes Escalas**:
    - Em problemas de regressão, diferentes conjuntos de dados podem ter variáveis alvo com escalas muito diferentes.
    - Normalizar y em um problema pode não ser apropriado em outro, tornando a normalização da variável alvo inconsistente.
- **Concentração nas Features (x)**:
    - A normalização, na maioria das vezes, se concentra em padronizar os atributos (X) para garantir que as métricas de distância considerem todos os atributos igualmente.
    - O foco está na consistência das escalas dos atributos, não na variável alvo.

#### Por que selecionar o número ideal de K em uma regressão com KNN?
- **Impacto no Desempenho**
- **Balanceamento entre Viés e Variância**
- **Busca pela Generalização Ideal**
    - O modelo vai generalizar bem para dados não vistos, fazendo previsões precisas.
- **Validação Cruzada como Ferramenta**
    - A seleção de hiperparâmetros, como K, geralmente é feita usando técnicas de validação cruzada, como K-fold.
- **Evitar Overfitting e Underfitting**
- **Impacto nas previsões e performance do modelo**

#### Método do Cotovelo (Elbow Method)

O Método do Cotovelo é uma técnica usada para determinar o número idela de clusters ou vizinhos em algoritmos de machine learning, como K-Means ou KNN.

![cotovelo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSFLIxdAauLJ3npj742mdce7CvBLhuuN2Ilp9_856Jg6g&s)

#### Distâncias para o KNN

Para calcular as diferentes distâncias entre os pontos (explicamos o porquê acima) o algoritmo pode usar diferentes métricas, algumas delas são: **Euclidiana**, **Manhattan** e **Minkowski**.

![distancias](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/1200px-Manhattan_distance.svg.png)

**Distância Euclidiana (verde)**
- Métrica mais comum.
- Calculada como a raiz quadrada da soma dos quadrados das diferenças entre as coordenadas de dois pontos.
- De maneira mais simples, calculando a diferença entre as coordenadas dos pontos em cada dimensão e, em seguida, aplicando o teorema de Pitágoras para obter a distância.

**Distância Manhattan (azul)**
- Também chamada de distância de cidade.
- É calculada como a soma das diferenças absolutas entre as coordenadas de dois pontos.

$$ d=|xa−xb|+|ya−yb|
$$

**Distância Minkowski (amarela p = 2) (vermelha p = 1)**

- A distância Minkowski, é uma métrica mais geral que engloba as distâncias Euclidiana e Manhattan. 
É definida como:
$$d_minkowski(x, y, p) = (Σ|xi - yi|^p)^(1/p)$$
- O parâmetro p determina o tipo de distância (p=1 e equivalente à distância Manhattan, p=2 é equivalente à distância Euclidiana). 
- Ela é útil quando queremos ajustar à sensibilidades nos dados.