import numpy as np
import pandas as pd

def gerar_base_alunos(n_alunos: int = 1000, seed: int = 42) -> pd.DataFrame:

    np.random.seed(seed)
    
    horas_estudo = np.random.normal(loc=14.0, scale=5.5, size=n_alunos)
    horas_estudo = np.clip(horas_estudo, 2.0, 35.0).round(1)
    
    frequencia = np.random.beta(a=5, b=1.5, size=n_alunos) * 50 + 50
    frequencia = np.clip(frequencia, 50.0, 100.0).round(1)
    
    base_ativ = (frequencia - 50) / 50 * 7 + np.random.normal(2.0, 1.5, size=n_alunos)
    atividades_entregues = np.clip(np.round(base_ativ), 0, 10).astype(int)

    nota_anterior = np.random.normal(loc=6.5, scale=1.8, size=n_alunos)
    nota_anterior = np.clip(nota_anterior, 0.0, 10.0).round(1)
    
    contrib_anterior = 0.35 * nota_anterior
    contrib_horas = 0.25 * (horas_estudo / 35.0 * 10.0)
    contrib_atividades = 0.20 * atividades_entregues
    contrib_frequencia = 0.20 * (frequencia / 100.0 * 10.0)
    
    ruido = np.random.normal(loc=0.0, scale=0.55, size=n_alunos)
    
    nota_final = contrib_anterior + contrib_horas + contrib_atividades + contrib_frequencia + ruido
    nota_final = np.clip(nota_final, 0.0, 10.0).round(1)
    
    df = pd.DataFrame({
        "id_aluno": range(1, n_alunos + 1),
        "horas_estudo": horas_estudo,
        "frequencia": frequencia,
        "atividades_entregues": atividades_entregues,
        "nota_anterior": nota_anterior,
        "nota_final": nota_final
    })
    
    return df

if __name__ == "__main__":
    df = gerar_base_alunos(n_alunos=1000, seed=42)
    caminho_csv = "dados_alunos.csv"
    df.to_csv(caminho_csv, index=False, encoding="utf-8")
    print(f"Base de dados sintética gerada com sucesso: '{caminho_csv}' ({len(df)} registros)")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print("\nEstatísticas descritivas básicas:")
    print(df.describe().round(2))
