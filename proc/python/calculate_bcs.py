b0 = float(input("Source buoyancy: "))
r0 = float(input("Source radius: "))
w0 = float(input("Source velocity: "))
alpha = float(input("Prescribed entrainment coefficient: "))

factor1 = 5/(6*alpha)
factor2 = (9/10 * alpha)**(-1/3)
factor3 = (5*r0/(6*alpha))**(-5/3)

f0 = (0.5*b0/(factor1*factor2*factor3))**(3/2) / r0**2
print("Buoyancy flux per unit area from b0: ",f0)

factor1 = 5/(6*alpha)
factor2 = (9/10 * alpha)**(1/3)
factor3 = (5*r0/(6*alpha))**(-1/3)

f0 = (0.5*w0/(factor1*factor2*factor3))**3 / r0**2
print("Buoyancy flux per unit area from w0: ",f0)

