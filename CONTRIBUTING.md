# Como contribuir

## Branches

`tipo/descricao-curta`, em minúsculas, com hífens e sem acentos.

| Tipo | Uso |
|---|---|
| `feat/` | nova funcionalidade |
| `fix/` | correção de bug |
| `hotfix/` | correção urgente em produção |
| `refactor/` | refatoração sem mudar comportamento |
| `docs/` | documentação |
| `test/` | testes |
| `perf/`, `style/`, `build/`, `ci/`, `chore/` | desempenho, formatação, build, CI, manutenção |
| `release/` | preparação de versão |

Exemplo: `feat/login-google`, `fix/45-erro-cadastro`.

## Commits ([Conventional Commits](https://www.conventionalcommits.org/pt-br/))

```
tipo(escopo opcional): descrição no imperativo

corpo opcional explicando o porquê

Closes #45
```

Exemplos: `feat(auth): adiciona login com Google`, `fix: corrige validação de CPF`.

## Fluxo

1. Crie a branch a partir da `main` atualizada.
2. Faça commits pequenos, no padrão.
3. `git push -u origin <branch>` e abra um Pull Request para a `main`.
4. O título do PR segue o padrão dos commits; o merge é feito por **squash**.

Push direto na `main` é bloqueado. O check `padrao-git` valida branch, título do PR e commits.
