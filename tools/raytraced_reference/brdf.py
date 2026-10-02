"""Independent scalar GGX oracle: Cycles' lambda formulation, not a fitted lobe."""
import math


def normalize(v):
    length = math.sqrt(sum(x*x for x in v))
    return tuple(x/length for x in v)


def specular(n, l, v, rough, f0):
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    nl,nv = dot(n,l),dot(n,v)
    if f0 <= 0 or nl <= 0 or nv <= 0:
        return 0.0
    h = normalize(tuple(a+b for a,b in zip(l,v)))
    nh,vh = dot(n,h),dot(v,h)
    a2 = rough**4
    distribution = a2/(math.pi*(1-nh*nh+nh*nh*a2)**2)
    smith = 2/(math.sqrt(1+a2*(1/(nl*nl)-1))+math.sqrt(1+a2*(1/(nv*nv)-1)))
    fresnel = f0+(1-f0)*(1-vh)**5
    return distribution*smith*fresnel/(4*nl*nv)


def cases():
    # Relative azimuth matters: N.L and N.V alone do not specify a microfacet BRDF.
    result=[]
    for tl,tv,phi in ((0,0,0),(30,30,180),(60,60,180),(70,20,60),(25,55,100),(80,75,170)):
        tl,tv,az=map(math.radians,(tl,tv,phi))
        for rough,f0 in ((0.12,0.04),(0.35,0.04),(0.65,0.65),(0.5,0.0)):
            l=(math.sin(tl),0,math.cos(tl))
            v=(math.sin(tv)*math.cos(az),math.sin(tv)*math.sin(az),math.cos(tv))
            result.append(dict(l=l,v=v,rough=rough,f0=f0))
    return result
