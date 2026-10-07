## RBG (Regressão Bem Geral)
*a.k.a Rodrigo do Bigode Grosso*

This is the official repository for the machine learning library intitled RBG.

This file contains the structure of the library, the workflow with git/github that should be used and the code style guides. This document should be referred if you have some doubt (it is algo a good context base for asking something to AI).

#### Git/Github workflow

Each feature should be developed in a separate branch, and no two people should work in the same branch or feature at the same time. The branches are meant to be incremental: don't create only one branch for all of your work. Instead, create a branch for every feature you want to implement, make a pull request to merge it into the main, and then delete it.

#### Basic git workflow

To make some change, you should first create a branch (use a suggestive name for the branch):

```bash
git branch -c my-new-branch
git switch my-new-branch
```

To make sure that your branch is up to date with the current main branch, do:

```bash
git pull origin main #this synchronizes your branch with the current main version on github
```

After making some changes, save them with:

```bash
git add . #the . adds all of the modified files
```

And then commit them with:

```bash
git commit -m "my commit message"
```

When you have the necessary commits, tested the new feature you have implemented, and are ready to merge them into the main branch, you should create a pull request. To do this, make:

```bash
git fetch #this takes the current data on github and puts it onto your pc
git merge origin/main #this merges the current version of the main branch into your feature branch
```

After doing this, you can open github, and an option to pull request will appear:

![Pull Request Example](images/pr.png)
