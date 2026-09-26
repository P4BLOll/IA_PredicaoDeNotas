import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gerar_dados import gerar_base_alunos

def main():
    # Carrega o arquivo com os dados dos alunos (ou cria se nao existir)
    caminho_csv = "dados_alunos.csv"
    if not os.path.exists(caminho_csv):
        print("Gerando base de dados dos alunos...")
        df = gerar_base_alunos(n_alunos=1000, seed=42)
        df.to_csv(caminho_csv, index=False, encoding="utf-8")
    else:
        df = pd.read_csv(caminho_csv)

    print(f"Total de alunos na base: {len(df)}")
    
    # Pasta onde os graficos serao salvos
    os.makedirs("graficos", exist_ok=True)

    # Variaveis de entrada e variavel alvo (nota final)
    features = ["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]
    target = "nota_final"

    X = df[features]
    y = df[target]

    # Separando 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Treinando a regressao linear
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Fazendo as previsoes no conjunto de teste
    y_pred = modelo.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 10.0)  # mantem as notas entre 0 e 10

    # Calculo dos residuos (erros das previsoes)
    erros = y_test.values - y_pred
    n_teste = len(y_test)

    # Medias e desvios padroes
    media_real = float(np.mean(y_test))
    desvio_real = float(np.std(y_test, ddof=1))

    media_pred = float(np.mean(y_pred))
    desvio_pred = float(np.std(y_pred, ddof=1))

    # Estatisticas dos erros
    media_erro = float(np.mean(erros))         # vies medio
    desvio_erro = float(np.std(erros, ddof=1)) # desvio padrao dos erros
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Intervalo de confianca para a media do erro (95%)
    # Para n=200 usamos t ~ 1.972
    erro_padrao = desvio_erro / np.sqrt(n_teste)
    margem_media = 1.972 * erro_padrao
    ic_media_inf = media_erro - margem_media
    ic_media_sup = media_erro + margem_media

    # Intervalo de erro individual (margem de +- 1.96 * desvio padrao)
    margem_individual = 1.96 * desvio_erro
    quantil_inf = float(np.percentile(erros, 2.5))
    quantil_sup = float(np.percentile(erros, 97.5))

    # Exibindo os resultados estatisticos no terminal
    print("\n" + "=" * 50)
    print("           RESULTADOS ESTATISTICOS")
    print("=" * 50)
    print(f"Media das Notas Reais:         {media_real:.3f} (Desvio: {desvio_real:.3f})")
    print(f"Media das Previsoes:          {media_pred:.3f} (Desvio: {desvio_pred:.3f})")
    print(f"Media do Erro (Vies):         {media_erro:.4f}")
    print(f"Desvio Padrao do Erro:        {desvio_erro:.4f}")
    print(f"Erro Medio Absoluto (MAE):    {mae:.4f}")
    print(f"Raiz Erro Quadratico (RMSE):  {rmse:.4f}")
    print(f"Coeficiente R2:               {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"IC 95% da Media do Erro:      [{ic_media_inf:.4f}, {ic_media_sup:.4f}]")
    print(f"Intervalo de Erro (95%):      +- {margem_individual:.3f} pontos (Quantis: [{quantil_inf:.2f}, {quantil_sup:.2f}])")
    print("=" * 50)

    print("\nEquacao ajustada:")
    print(f"Nota = {modelo.intercept_:.4f}", end="")
    for feat, coef in zip(features, modelo.coef_):
        print(f" + ({coef:.4f} * {feat})", end="")
    print("\n")

    # --- GERACAO DOS GRAFICOS ---
    print("Gerando os graficos...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Matriz de correlacao
    plt.figure(figsize=(7, 5))
    corr = df[features + [target]].corr()
    im = plt.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, label="Correlacao de Pearson")
    ticks = range(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=25, ha="right")
    plt.yticks(ticks, corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if corr.iloc[i, j] < 0.7 else "white")
    plt.title("Matriz de Correlacao Linear", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graficos/1_correlacao_features.png", dpi=300)
    plt.close()

    # 2. Notas Reais vs Notas Previstas
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#2563eb", edgecolors="black", linewidth=0.5, label="Alunos do Teste")
    min_v, max_v = min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5
    eixo = np.linspace(min_v, max_v, 100)
    plt.plot(eixo, eixo, color="#dc2626", linestyle="--", linewidth=2, label="Linha Ideal (y = x)")
    plt.fill_between(eixo, eixo - margem_individual, eixo + margem_individual, color="#dc2626", alpha=0.12, label=f"Faixa de 95% (+- {margem_individual:.2f} pts)")
    plt.xlabel("Nota Real")
    plt.ylabel("Nota Prevista")
    plt.title(f"Notas Reais vs Previstas (R2 = {r2:.3f})", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graficos/2_real_vs_predito.png", dpi=300)
    plt.close()

    # 3. Distribuicao dos erros
    plt.figure(figsize=(8, 5))
    plt.hist(erros, bins=25, density=True, alpha=0.6, color="#16a34a", edgecolor="black", label="Erros Observados")
    x_curva = np.linspace(min(erros) - 0.5, max(erros) + 0.5, 200)
    curva_normal = (1 / (desvio_erro * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_curva - media_erro) / desvio_erro) ** 2)
    plt.plot(x_curva, curva_normal, "k--", linewidth=2, label=f"Normal (media={media_erro:.2f}, desvio={desvio_erro:.2f})")
    plt.axvline(media_erro, color="red", linestyle="-", label=f"Media ({media_erro:.2f})")
    plt.axvline(media_erro - margem_individual, color="purple", linestyle=":", label=f"-1.96 desvios ({quantil_inf:.2f})")
    plt.axvline(media_erro + margem_individual, color="purple", linestyle=":", label=f"+1.96 desvios ({quantil_sup:.2f})")
    plt.xlabel("Erro da Previsao (Real - Previsto)")
    plt.ylabel("Densidade")
    plt.title("Distribuicao dos Erros e Intervalo de 95%", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graficos/3_distribuicao_erros.png", dpi=300)
    plt.close()

    # 4. Residuos vs Previsoes (homocedasticidade)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, erros, alpha=0.6, color="#7c3aed", edgecolors="black", linewidth=0.5)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.8, label="Erro Zero")
    plt.axhline(margem_individual, color="gray", linestyle=":", label=f"+1.96 DP ({margem_individual:.2f})")
    plt.axhline(-margem_individual, color="gray", linestyle=":", label=f"-1.96 DP (-{margem_individual:.2f})")
    plt.xlabel("Nota Prevista")
    plt.ylabel("Residuo (Real - Previsto)")
    plt.title("Residuos vs Previsoes", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graficos/4_residuos_vs_predicoes.png", dpi=300)
    plt.close()

    # 5. Importancia das variaveis (coeficientes)
    plt.figure(figsize=(8, 4.5))
    nomes = ["Horas de Estudo", "Frequencia", "Atividades Entregues", "Nota Anterior"]
    pesos = [modelo.coef_[i] for i in range(len(features))]
    y_pos = range(len(nomes))
    plt.barh(y_pos, pesos, color=["#2563eb", "#16a34a", "#ea580c", "#dc2626"], edgecolor="black")
    plt.yticks(y_pos, nomes)
    for i, v in enumerate(pesos):
        plt.text(v + 0.005, i, f"+{v:.4f}", va="center", fontweight="bold")
    plt.xlabel("Peso / Coeficiente na Nota Final")
    plt.title("Impacto de Cada Variavel no Modelo", fontsize=13, fontweight="bold")
    plt.xlim(0, max(pesos) * 1.25)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("graficos/5_importancia_variaveis.png", dpi=300)
    plt.close()

    print("Todos os graficos foram gerados e salvos na pasta 'graficos/'.")

if __name__ == "__main__":
    main()
