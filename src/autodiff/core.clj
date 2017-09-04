(ns autodiff.core
  (:refer-clojure :exclude [* + - /])
  )

(defprotocol AutoDiff
  (add [u v])
  (sub [u v])
  (mul [u v])
  (negate [u])
  (to-num [u])
  )

(defmacro destruct
  [content]
  `(let ~[{'u :f 'u' :f' :or {'u 'u 'u' 0}} 'u
          {'v :f 'v' :f' :or {'v 'v 'v' 0}} 'v]
     ~content))


(defrecord Dual
    [f f']
  AutoDiff
  (add [u v]
    (destruct
      (Dual. (add u v) (add u' v'))))
  (sub [u v]
    (destruct
      (Dual. (sub u v) (sub u' v'))))
  (mul [u v]
    (destruct
      (Dual. (mul u v) (add (mul u' v) (mul u v')))))

  ;; (negate [u] (update (Dual. 6 3) :f clojure.core/-))
  )

(defn coerce
  "Makes value a Dual if not already"
  [x]
  (if (= (str (type x)) "class autodiff.core.Dual")
    x (->Dual x 0)))


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
    (add (coerce a) (coerce b))))
 (sub [a b]
      (if (and (number? a) (number? b))
        (clojure.core/- a b)
        (sub (coerce a) (coerce b))))
 (mul [a b]
      (if (and (number? a) (number? b))
        (clojure.core/* a b)
        (mul (coerce a) (coerce b))))
 )



(defn exp [n pow]
  (reduce mul (repeat pow n)))


(defn d
  "Find the first derivative of a function"
  [f & args]
  (apply f (map #(Dual. % 1) args)))




;; Math operators

(defn +
  "Replace clojure.core/+"
  ([] 0)
  ([a] a)
  ([a b]
   (if (and (number? a) (number? b))
     (clojure.core/+ a b)
     (add a b)))
  ([a b & more]
   (reduce + (+ a b) more)))


(defn -
  "Replace clojure.core/-"
  ([a] (negate a))
  ([a b]
   (if (and (number? a) (number? b))
     (clojure.core/- a b)
     (sub a b)))
  ([a b & more]
   (reduce - (- a b) more)))



(defn *
  "Replace clojure.core/-"
  ([a] (negate a))
  ([a b]
   (if (and (number? a) (number? b))
     (clojure.core/* a b)
     (mul a b)))
  ([a b & more]
   (reduce * (* a b) more)))
