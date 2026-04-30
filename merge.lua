import("csvfast", "json", "cml", "cstats", "task", "Table")

-- SHARD DISCOVERY (Se mantiene igual)
local function list_files(prefix)
    local p = io.popen("ls data/" .. prefix .. "_*.csv 2>/dev/null")
    if not p then return {} end
    local files = {}
    for file in p:lines() do
        local n = file:match(prefix .. "_(%d+)%.csv")
        if n then files[#files+1] = { path = file, idx = tonumber(n) } end
    end
    p:close()
    table.sort(files, function(a, b) return a.idx < b.idx end)
    return files
end

local function bool(v)
    if v == nil then return 0 end
    local s = string.lower(tostring(v))
    if s == "1" or s == "t" or s == "true" or s == "yes" then return 1 end
    return 0
end

local train_shards = list_files("dataset_train")
local test_shards  = list_files("dataset_test")

-- STORAGE & SCHEMA (Se mantiene igual)
local train, test = {}, {}
local schema = {
    log_price = true, accommodates = true, bathrooms = true, bedrooms = true,
    beds = true, reviews = true, rating = true, lat = true, lon = true,
    cleaning_fee = true, instant_bookable = true, host_verified = true,
    room_entire = true, room_private = true, room_shared = true, is_apartment = true
}

local function split_dataset(data, ratio)
    local n = #data
    if n < 2 then return data, {} end

    --[[for i = n, 2, -1 do
        local j = math.random(i)
        data[i], data[j] = data[j], data[i]
    end]]
    Table.shuffle(data)    

    local split_idx = math.floor(n * ratio)
    local train_set = {}
    local test_set  = {}

    for i = 1, n do
        if i <= split_idx then
            train_set[#train_set + 1] = data[i]
        else
            test_set[#test_set + 1] = data[i]
        end
    end

    return train_set, test_set
end

local function loader(shards, target, label)
    for _, s in ipairs(shards) do
        print("["..label.."] loading:", s.path)
        csvfast.each(s.path, function(r)
            if not r then return end
            local y = r.log_price
            if not y or y ~= y then return end
            local lat = r.lat or r.latitude
            local lon = r.lon or r.longitude
            if not lat or not lon then return end

            target[#target+1] = {
                log_price = y,
                accommodates = r.accommodates, bathrooms = r.bathrooms,
                bedrooms = r.bedrooms, beds = r.beds, reviews = r.reviews,
                rating = r.rating, lat = lat, lon = lon,
                cleaning_fee = r.cleaning_fee,
                instant_bookable = bool(r.instant_bookable),
                host_verified = bool(r.host_verified),
                room_entire = bool(r.room_entire),
                room_private = bool(r.room_private),
                room_shared = bool(r.room_shared),
                is_apartment = bool(r.is_apartment)
            }
        end, { schema = schema })
        task.wait(0)
    end
end

task.spawn(loader, train_shards, train, "TRAIN")
task.spawn(loader, test_shards, test, "TEST")
while task.step() do end

print("Train size inicial:", #train)
print("Test size inicial :", #test)

if #test == 0 then
    print("[WARN] No se detectaron archivos de test. Realizando split automatico (80/20)...")
    train, test = split_dataset(train, 0.8)
end

print("Final Train size:", #train)
print("Final Test size :", #test)

blund(not(#test == 0 or #train == 0), "Error: Conjuntos de datos vacios. Revisa la ruta de los CSV.")

-- FEATURES & MODEL
local features = {"accommodates", "bathrooms", "bedrooms", "beds", "reviews", "rating", "lat", "lon", "cleaning_fee", "instant_bookable", "host_verified", "room_entire", "room_private", "room_shared", "is_apartment"}

local model = cml.LinearRegression({
    features = features,
    target = "log_price"
})

print("[MODEL] Fitting...")
model:load(train)
model:normalize()
model:fit(train, 0.05, 5000, 0.8, 0.02)

-- 1. Extraer vectores de la tabla test
local t_true, t_pred = {}, {}
for i = 1, #test do
    t_true[i] = test[i].log_price
    t_pred[i] = model:predict(test[i])
end

-- 2. Convertir a objetos cstats.array (Memoria C contigua)
local y_true = cstats.array(t_true)
local y_pred = cstats.array(t_pred)

-- 3. Calcular métricas usando la nueva API de objetos
-- Nota: Usamos los objetos y_true/y_pred directamente
local mae = y_true:mae(y_pred)
local mse = y_true:mse(y_pred)
local r2  = y_true:r2(y_pred)
local corr = y_true:corr(y_pred)

print("===== METRICS (C STATS API) =====")
print("Address y_true:", y_true) -- Verificamos el nuevo __tostring
print(string.format("MAE : %.6f", mae))
print(string.format("MSE : %.6f", mse))
print(string.format("R2  : %.6f", r2))
print(string.format("Corr: %.4f", corr))

-- EXPORTACION
local out = {
    meta = { train = #train, test = #test, features = features },
    metrics = { mae = mae, mse = mse, r2 = r2, corr = corr },
    model = model:export()
}

local f = assert(io.open("data/model_final.json", "w"))
f:write(json.encode(out))
f:close()

print("MODEL OK -> data/model_final.json")
