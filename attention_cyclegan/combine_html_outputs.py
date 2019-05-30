from pathlib import Path

base = """
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>Attention-Guided CG</title>
    <script src="https://code.jquery.com/jquery-3.4.1.min.js"
        integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=" crossorigin="anonymous"></script>
    <style>
    <style>
        html,
        body {
            height: 100%;
            max-height: 100%;
            margin: 0px;
        }

        object {
            width: 400px;
            height: 200px;
        }

        iframe {
            width: 100%;
            height: 90%;
        }
        #goUp {
            position: fixed;
            bottom: 10;
            right: 10;
            background: lightskyblue;
            padding: 14px 20px 6px;
            border-radius: 109px;
            font-size: 2em;
            color: white;
            cursor: pointer;
        }
        img {
            margin: 5px;
        }
    </style>
</head>

<body>

<div id="goUp">^</div>

<div id="control">
    <form action="">
        <label for="epoch">Epoch: &nbsp;</label><input type="number" name="epoch" id="epochInput" value="0" />
        <label for="domains">Domain: &nbsp;</label><select name="domains" id="selectDomain">
            <option value="AB">AB</option>
            <option value="A">A</option>
            <option value="B">B</option>
        </select>
        <button type="submit">Go</button>
    </form>
</div>

<h3 id="loading">loading...</h3>

<div id="imgs" style='display:none'>

"""

end = """
</div>
</body>

<script>
$(function () {
    $("#goUp").click(() => {
        $(window).scrollTop(0);
    })

    $("#loading").hide()

    $("form").submit((e) => {
        $("#loading").show();
        $("#imgs").show()
        e.preventDefault()
        // alert($("#epochInput").val() + " " + )
        const domain = $("#selectDomain option:selected").text();
        const epoch = parseInt($("#epochInput").val(), 10);
        $("img").hide()
        $(".imgDiv").hide()
        $("#imgs" + epoch).show()
        console.log("#imgs" + epoch)
        $("img").each((i, el) => {
            let src = $(el).attr("src");
            const ep = parseInt(src.match(/\d+/)[0], 10);
            let dom;
            if (domain !== "AB"){
                if (src.indexOf("mask_") >= 0) {
                    dom = src.split("_")[1].indexOf(domain.toLowerCase())
                } else {
                    dom = src.split("_")[0].indexOf(domain);
                }
            } else {
                dom = 0;
            }
            console.log(ep, dom)
            if (ep === epoch && dom >= 0) {
                $(el).show()
            }
        })
        $("#loading").hide();
    })
});
</script>

</html>
"""

if __name__ == "__main__":

    # 1. put this file alongside imgs/ and epoch_X.html files
    # 2. run it $ python combine.py
    # 3. open combined.html
    # 4. wait a bit, there are a lot of images to load. When it's done,
    #    the "loading..." text will disapear
    # 5. select epoch number and domains (A/B/A and B)
    # 6. click "go" or hit return key

    files = Path().glob("*epoch*.html")
    final = "<h1>Epochs</h1>"
    files = sorted(files, key=lambda x: x.name)
    for i, fi in enumerate(files):
        with fi.open("r") as f:
            id = fi.name.split("epoch_")[1].split(".html")[0]
            final += f"<div style='display:none' class='imgDiv' id='imgs{id}'><h2 id='head{id}' class='epochHead'>Epoch {id} </h2>\n {f.read()}</div>"
            # if i > 10:
            #     break
    with open("combined.html", "w") as f:
        f.write(base + final + end)
