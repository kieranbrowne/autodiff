(ns autodiff.core
  (:refer-clojure :exclude [* + - /])
  (:require [autodiff.protocols :as ad :refer [AutoDiff ->Dual]])
  )



(defmacro extend-types [ts & specs]
  (conj (into '() (map (fn [type]
                         (into specs [type 'extend-type]))
                       ts)) 'do))

(extend-types
 [java.lang.Number
  java.lang.Long
  java.lang.Double]

 AutoDiff

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
 (negate [a] (clojure.core/- a))
 (sin [a] (Math/sin a))
 (cos [a] (Math/cos a))
 )

(Math/sin 2)


(defn d
  "Find the first derivative of a function"
  [f & args]
  (:f'
   (apply f (map #(->Dual % 1) args))))



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
