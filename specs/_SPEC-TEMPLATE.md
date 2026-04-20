# [Nom du module] — Specification

> **Statut :** DRAFT | READY | IMPLEMENTED | DEPRECATED
> **Version :** 1.0
> **Niveau :** 1 (leger) | 2 (standard) | 3 (exhaustif)
> **Auteur :** [prenom nom]
> **Date de creation :** [YYYY-MM-DD]

<!-- GUIDE DES NIVEAUX
  Niveau 1 (leger)   : Sections 1, 3, 4, 6 obligatoires. Le reste optionnel.
  Niveau 2 (standard): Toutes les sections obligatoires.
  Niveau 3 (exhaustif): Toutes les sections + decomposition multi-fichiers.
-->

---

## 1. Objectif

<!-- OBLIGATOIRE tous niveaux. 1-3 phrases. Pourquoi ce module existe. Quel probleme il resout. -->

## 2. Perimetre

<!-- Niveau 2+ obligatoire. Niveau 1 optionnel mais recommande. -->

### IN — Ce que le module fait

- ...

### OUT — Ce que le module ne fait PAS

<!-- Aussi important que le IN. Evite le scope creep. -->

- ...

---

## 3. Interface

<!-- OBLIGATOIRE tous niveaux. -->

### Fonction(s) principale(s)

```
nomDeLaFonction(param1: Type, param2: Type): ReturnType
```

### Inputs

| Param | Type | Obligatoire | Description | Valeur par defaut | Exemple |
|-------|------|:-----------:|-------------|-------------------|---------|
| | | | | | |

### Outputs

| Champ | Type | Description | Exemple |
|-------|------|-------------|---------|
| | | | |

### Erreurs

| Code / Type | Condition de declenchement | Message / Comportement |
|-------------|---------------------------|------------------------|
| | | |

---

## 4. Regles metier

<!-- OBLIGATOIRE tous niveaux. Chaque regle = un test. Numerotation stable (ne pas renumeroter si suppression). -->

- **R1:** [Description de la regle]
  - *Ref :* [source normative, lien, doc interne — si applicable]

- **R2:** [...]

---

## 5. Edge cases

<!-- Niveau 2+ obligatoire. Situations limites, inputs degrades, cas rares mais possibles. -->

- **EC1:** [Situation] → [Comportement attendu]
- **EC2:** [...]

---

## 6. Exemples concrets

<!-- OBLIGATOIRE tous niveaux. -->

### Cas nominal

```
Input:  [...]
Output: [...]
```

### Cas d'erreur

```
Input:  [...]
Output: [erreur attendue]
```

<!-- Ajouter autant d'exemples que necessaire pour lever toute ambiguite. -->

---

## 7. Dependances & contraintes

<!-- Niveau 2+ obligatoire. -->

### Techniques

- Runtime : [Node >= 20 / Python >= 3.11 / ...]
- Module system : [ESM / CJS / ...]
- Dependances externes : [liste ou "aucune"]

### Performance

- [ex: < 2s pour 10k lignes d'entree]

### Securite

- [ex: pas de donnees client en clair dans les logs]

---

## 8. Changelog

| Version | Date | Modification |
|---------|------|--------------|
| 1.0 | YYYY-MM-DD | Creation initiale |
