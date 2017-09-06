(ns autodiff.protocols)

(defprotocol AutoDiff
  (add [u v])
  (sub [u v])
  (mul [u v])
  (negate [u])
  (to-num [u])
  (pi [u])
  (exp [u])
  (sqrt [u])
  (log [u])
  (sin [u])
  (cos [u])
  (tan [u])
  (asin [u])
  (acos [u])
  (atan [u])
  (sinh [u])
  (cosh [u])
  (tanh [u])
  (asinh [u])
  (acosh [u])
  (atanh [u]))

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


(defn coerce
  "Makes value a Dual if not already"
  [x]
  (if (= (str (type x)) "class autodiff.protocols.Dual")
    x (->Dual x 0)))
