(ns autodiff.core
  (:refer-clojure :exclude [* + - /]))

(defprotocol AutoDiff
  (+ [u v])
  (- [u v])
  (* [u v])
  )

(defmacro destruct
  [content]
  `(let ~[{'u :f 'u' :f' :or {'u 'u 'u' 0}} 'u
          {'v :f 'v' :f' :or {'v 'v 'v' 0}} 'v]
     ~content))


(defrecord Dual [f f']
  AutoDiff
  (+ [u v]
    (destruct
      (Dual. (+ u v) (+ u' v'))))
  (- [u v]
    (destruct
      (Dual. (- u v) (- u' v'))))
  (* [u v]
    (destruct
      (Dual. (* u v) (+ (* u' v) (* u v')))))
  )

(extend-type java.lang.Long
  AutoDiff
  (+ [u v] (clojure.core/+ u v))
  (- [u v] (clojure.core/- u v))
  (* [u v] (clojure.core/* u v))
    )
