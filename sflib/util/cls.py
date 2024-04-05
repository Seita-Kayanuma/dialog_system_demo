# class関係のユーティリティ
def construct(module_name, class_name, ctor_args=None):
    """モジュール名，クラス名，コンストラクタへの引数を与えて
    インスタンスを作成する．

    Args:
      module_name(str): モジュール名
      class_name(str): クラス名
      ctor_args(tuple, list, dict or None): コンストラクタに与える引数をまとめたもの．
        tuple が与えられた場合は，ctor_args[0]が順序引数，
              ctor_args[1]が名前付き引数として扱われる．
        list が与えられた場合は順序引数，
        dict が与えらえれた場合は名前付き引数，
        Noneの場合は引数が与えられずに構築される．
    
    Returns:
      構築したインスタンス
    """
    c_args, c_kwargs = [], {}
    if isinstance(ctor_args, tuple):
        c_args, c_kwargs = ctor_args
    elif isinstance(ctor_args, list):
        c_args = ctor_args
    elif isinstance(ctor_args, dict):
        c_kwargs = ctor_args
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(*c_args, **c_kwargs)
