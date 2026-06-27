# Technití-Ánoia

<sub>*Nota de Transparência: Este arquivo README.md foi gerado e refinado com o auxílio do **Deep Seek** para garantir correções ortográficas e estruturais.*</sub>

---

## Hefesto e a Origem da Máquina Pensante

Hefesto, o ferreiro divino do Olimpo, era o deus grego do fogo, da metalurgia e da criação artesanal. Ele forjou servas de ferro — autômatos inteligentes — para servir aos deuses. Com seu martelo e sua inteligência, deu vida ao inanimado.

> “Hefesto era o artesão supremo, criador de objetos mágicos e autômatos que possuíam vida própria.”
> — Robert Graves, *The Greek Myths*

O nome **Technití Ánoia** (Τεχνητή Ἄνοια) refere-se à tolice artificial da humanidade moderna, imersa na tecnologia. Perdemos nossos costumes em troca do vício em telas; trocamos o pensamento profundo por respostas geradas.

---

## Sobre o Projeto

Este repositório contém uma implementação de componentes essenciais de um **Transformer**, baseada no artigo:

> **"Attention Is All You Need"**
> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
> *arXiv:1706.03762* (2017)

Em fins de 2019, ainda como um aprendiz autodidata (sem conhecimento sólido de C e praticamente sem inglês), traduzi o paper e implementei a primeira versão do modelo em C. Anos depois, recuperei o código de um pen drive antigo e estou reconstruindo e evoluindo tudo em **Zig 0.16.0**.

O objetivo é criar um motor de IA leve, compreensível, bem instrumentado e otimizado, priorizando clareza e performance.

### Funcionalidades Atuais

- Operações básicas de álgebra linear (matmul com blocking, transpose, scale)
- Funções de ativação (Sigmoid, ReLU, Leaky ReLU, Softmax)
- RMS Normalization
- Máscara causal para atenção
- **Multi-Head Attention** completo
- Sistema de telemetria e análise de tensores (média, norma L2, densidade/esparsidade)
- Visualização de densidade com barras de progresso

---

## Estrutura do Projeto

```bash
src/
├── main.zig
├── transformer-c/
│   └── transformer.c           # Meu código original (2019)
├── test/
│   └── test.zig
├── analyzer/
│   ├── analyze.zig
│   └── printbar.zig
├── math/
│   ├── linalg.zig
│   └── ops.zig
└── nn/
    └── activation.zig
```

---

## Como Compilar e Executar
```bash
  zig build run
```

##  Resultado Esperado
```bash
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
TELEMETRIA
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
  ops.transpose          | Média:  3.500 | Norma L2:  9.539 | Zeros:   0.0% | Ok
  ops.matmul             | Média: 103.750 | Norma L2: 224.715 | Zeros:   0.0% | Ok
  ops.scale              | Média: 51.875 | Norma L2: 112.358 | Zeros:   0.0% | Ok
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
  activation.sigmoid_f   | Média:  0.553 | Norma L2:  1.408 | Zeros:   0.0% | Ok
  activation.relu_f      | Média:  0.900 | Norma L2:  3.354 | Zeros:  60.0% | Ok
  activation.sigmoid_b   | Média:  0.157 | Norma L2:  0.391 | Zeros:   0.0% | Ok
  activation.relu_b      | Média:  0.400 | Norma L2:  1.414 | Zeros:  60.0% | Ok
  activation.softmax     | Média:  0.333 | Norma L2:  0.709 | Zeros:   0.0% | Ok
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
  neural.multi_head_attn | Média:  0.544 | Norma L2:  2.759 | Zeros:  12.5% | Ok
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
DENSIDADE E ESPARSIDADE DO TENSOR
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
  ops.transpose          | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] Escassez:   0.0%
  ops.matmul             | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] Escassez:   0.0%
  ops.scale              | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] Escassez:   0.0%
________________________________________________________
  activation.sigmoid_f   | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] Escassez:   0.0%
  activation.relu_f      | Densidade [▓▓▓▓▓▓▓▓            ] Escassez:  60.0%
  activation.sigmoid_b   | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] Escassez:   0.0%
  activation.relu_b      | Densidade [▓▓▓▓▓▓▓▓            ] Escassez:  60.0%
  activation.softmax     | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] Escassez:   0.0%
________________________________________________________
  neural.multi_head_attn | Densidade [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ] Escassez:  12.5%
________________________________________________________
```

---

## Evolução
O projeto está em desenvolvimento ativo.

---

**Autor:** Érison Cleyton

**Citação utilizada:**
Robert Graves, *The Greek Myths* (edição completa, 1955 / revisada posteriormente). É uma das obras mais respeitadas e completas sobre mitologia grega.
