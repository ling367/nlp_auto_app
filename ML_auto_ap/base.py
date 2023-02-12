# =============================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug 17 16:20:20 2022
# 
# @author: ling
# """
# #Les bases:
# #Arithmetique 
# #+,-,*,/,**
# #Comparaison 
# #<=,>=,==,!=,<,>
# #la réponse - true ou false (boolean)
# 
# 
# #and : &
# #or : |
# #xor : 
#     
#     
# #String (chaine des caractères)
# 
# #pour créer deux variable simultanément    x,y = 1,2
# 
# 
# #-----------------------------------------------------------------------
# #fonction du type f(x) = x**2
# 
# #la fonction : lambda 
# f = lambda x : x**2 
# print(f(2))
# f1 = lambda x,y : x**2 +y
# 
# #la définition de notre fonction 
# def e_protentielle(masse,hauteur,g,e_limite):
#     E = masse *hauteur*g
#     if E <= e_limite:
#         return True
#     else:
#         return False
# 
# resultat = e_protentielle(masse=1, hauteur=5, g = 9, e_limite= 10)
# print (resultat)
# 
# #if elif else
# print (resultat)
# 
# #liste 
# 
# #une liste peut contenir des liste
# # =============================================================================
# # liste1 = [1,2,4,6]
# ville = ['Paris', 'Londre','Pékin','Berlin']
# # liste_g = [liste1,ville]
# # print (liste_g)
# # liste_vide = []
# #liste[index]
# 
# #pour accéder au dernière élément : ville[-1] 
# #avant dernier ville[-2]
# # =============================================================================
# 
# #tuple 
# tuple_1 = (1,2,8,7) #une fois créé, il est protégé, on peut pas motifier 
# #Quand il y a bcp bcp de donnée, tuple plue rapide à exécuter 
# 
# #string  - chaîne de caractères 
# 
# #séquence = une structure des données en ordre 
# #chaque élément a un index = son numéro 
# 
# 
# #technique de SLICING 
# 
# #liste [début:fin:pas] #la fin n'est pas compris
# #print à l'envers la liste 
# print (ville[::-1])
# print (ville[0:3])
# =============================================================================


#dictionnaire  - clé , les clés sont uniques 
#un ensemble d'affectation (clé:valeur)

dico_fr_ag = {
    "chien":"dog",
    "chat":"cat",
    "rat":"rat"
}


dico = {
    "bananas":10,
    "pommes":20,
    "poires":22
}

#on peut aussi nester les dico sans un dico

dico_nested = {
    "dico_fr_ag": dico_fr_ag,
    "dico": dico
    }


#dictionnaire dans machine learning : 
import numpy as np
#stoker les paramètres de leur réseaux de neurone 
paramètre ={
    "W1" : np.random.randn(10,100)
    }






























