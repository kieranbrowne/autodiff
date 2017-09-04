(ns autodiff.core
  ;; (:refer-clojure :exclude [* + - /])
  )

(defprotocol AutoDiff
  (add [u v])
  (sub [u v])
  (mul [u v])
  (const [u])
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
  )

(extend-type java.lang.Number
  AutoDiff
  (add [u v] (clojure.core/+ u v))
  (sub [u v] (clojure.core/- u v))
  (mul [u v] (clojure.core/* u v))
  (const [u] (Dual. u 1))
    )


(defn * [& args] (reduce mul args))
(defn + [& args] (reduce add args))
(defn - [& args] (reduce sub args))

(defn exp [n pow]
  (reduce mul (repeat pow n)))

(defn d
  "Find the first derivative of a function"
  [f & args]
  (apply f (map #(Dual. % 1) args)))
