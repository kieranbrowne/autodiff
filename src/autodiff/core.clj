(ns autodiff.core
  (:refer-clojure :exclude [* + - /])
  (:require [autodiff.protocols :as ad :refer [AutoDiff ->Dual]])
  )



(defmacro extend-types [ts & specs]
  (conj (into '() (map (fn [type]
                         (into specs [type 'extend-type]))
                       ts)) 'do))

(defn deep-map 
  "Apply function recursively until nodes are values other than collections are reached"
  [fn coll]
  (if (coll? coll) 
    (map (partial deep-map fn) coll) 
    (fn coll)))

(defn deep-mapv
  "Apply function recursively until nodes are values other than collections are reached"
  [fn coll]
  (if (coll? coll) 
    (mapv (partial deep-map fn) coll) 
    (fn coll)))

(extend-types
 [java.lang.Number
  java.lang.Long
  java.lang.Double]

 AutoDiff

 (constant [a] a)
 (add [a b]
      (if (and (number? a) (number? b))
        (clojure.core/+ a b)
        (ad/add (ad/coerce a) (ad/coerce b))))
 (sub [a b]
      (if (and (number? a) (number? b))
        (clojure.core/- a b)
        (ad/sub (ad/coerce a) (ad/coerce b))))
 (mul [a b]
      (if (and (number? a) (number? b))
        (clojure.core/* a b)
        (ad/mul (ad/coerce a) (ad/coerce b))))
 (div [a b]
      (if (and (number? a) (number? b))
        (clojure.core// a b)
        (ad/mul (ad/coerce a) (ad/coerce b))))
 (pow [a b]
      (if (and (number? a) (number? b))
        (Math/pow a b)
        (ad/pow (ad/coerce a) (ad/coerce b))))
 (log [a] (Math/log a))
 (tanh [a] (Math/tanh a))
 (exp [a] (Math/exp a))
 (sigmoid [a] (ad/div 1 (ad/add 1 (ad/pow (ad/exp 1) (ad/negate a)))))
 (negate [a] (clojure.core/- a))
 (sin [a] (Math/sin a))
 (cos [a] (Math/cos a))
 (transpose [a] a)
 (pi [a] Math/PI)
 (val-like [a v] v)
 (one [a] 1.)
 (two [a] 2.)
 (identity [a] a))
 

; (extend-types
;  [clojure.lang.PersistantVector]

;  AutoDiff

;  (constant [a] a)
;  (add [a b]
;       (if (number? b)
;         (deep-map (partial ad/add b) a)
;         (deep-map)))
;  (identity [a] a))
 


;; (Math/exp 1)

;; (ad/sigmoid 9)

;; (ad/pow 2 2)

;; (ad/d ad/add 1 2)
(ad/d ad/pow (ad/coerce 3 1) (ad/coerce 2 0))
;; (ad/pow 3 2)

(let [x (ad/constant 4)
      y (ad/constant 8)
      z (reduce ad/mul (repeat y x))]
  (ad/d ad/pow (ad/->Dual x 1) y)
  )

(Math/pow 2 2)
(ad/pow (ad/coerce 2) 2)
(ad/coerce 3)


;; Math operators

(defn +
  "Replace clojure.core/+"
  ([] 0)
  ([a] a)
  ([a b]
   (if (and (number? a) (number? b))
     (clojure.core/+ a b)
     (ad/add a b)))
  ([a b & more]
   (reduce + (+ a b) more)))


(defn -
  "Replace clojure.core/-"
  ([a] (ad/negate a))
  ([a b]
   (if (and (number? a) (number? b))
     (clojure.core/- a b)
     (ad/sub a b)))
  ([a b & more]
   (reduce - (- a b) more)))


(defn *
  "Replace clojure.core/-"
  ([a] a)
  ([a b]
   (if (and (number? a) (number? b))
     (clojure.core/* a b)
     (ad/mul a b)))
  ([a b & more]
   (reduce * (* a b) more)))
