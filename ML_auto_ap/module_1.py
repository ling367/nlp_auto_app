# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Les bases:
#Arithmetique 
#+,-,*,/,**
#Comparaison 
#<=,>=,==,!=,<,>
#la réponse - true ou false (boolean)


#and : &
#or : |
#xor : 
    
    
#String (chaine des caractères)

#pour créer deux variable simultanément    x,y = 1,2


#-----------------------------------------------------------------------
#fonction du type f(x) = x**2

#la fonction : lambda 
# =============================================================================
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
# =============================================================================




def fibo(n):
    a,b = 0,1
    fib = []
    while a<n:
        fib.append(a)
        a,b = b, a+b
    return fib



classeur = {
    'positif':[],
    'negatif':[]  
    }

def classer(classeur,nb):
    if nb >0:
        classeur['positif'].append(nb)
    else:
        classeur['negatif'].append(nb)
    return classeur




































